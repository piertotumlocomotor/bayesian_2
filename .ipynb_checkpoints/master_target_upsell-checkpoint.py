from re import search
from aa_engine_pkg.assets.utils import *
from aa_engine_pkg.assets.core.data.kedro.catalog_expansion.partitioned_sql import SQLPartitionedDataSet

def create_target_upsell(upgrades_basicos: SQLPartitionedDataSet,
                         eop: SQLPartitionedDataSet,
                         cliente_activo: pd.DataFrame,
                         parameters: Dict,
                         date: str) -> pd.DataFrame:
    """Function that takes care of generating the target feature for the up-sell model
    Up-selling includes:
        - Switching to a higher level of service on the same tecnology (e.g., from Silver SD to Gold SD)

    Target definition:
        - Existing customer switches service.
        - Stays with new product for at least 84 days

    Target methodology:
        0. Definition of product ranking (update as needed)
        1. Loading of customer base (e.g., all post-paid customers in Colombia)
        2. Loading of upgrade events (108) for period of interest (calculation_window)
        3. Loading of events (107,108,133,142) to detect customers that switch products for period of
        interest (activation_window)
        4. For customers that have an event (108) in the calculation_window, compare previous service to new one to determine if it
    is an upgrade using the product ranking.
        5. For said customers in (4), check if another event happens in the activation_window after the 108 event. If it does not happen, then the
    customers are target for the model.

    Parameters
    ----------
    upgrades_basicos: 
        dataset defined in ``catalog_raw.yml`` with raw data information related to upgrades of programming service products
    eop:
        dataset defined in ``catalog_raw.yml`` with raw data information related to the client's EoP state
    date:
        period to process
    parameters:
        set of project parameters defined in ``parameters.yml``
    
    Returns
    -------
    pd.DataFrame
        Master table with up-sell target feature for one period (date+1; date+calculation_window)
    """

    # Initialize logger
    log = initialize_logger()
    
    table_name = "target_upsell"
    write_to_parquet = parameters["write_to_parquet"]
    overwrite = parameters["targets"][table_name]["overwrite"]
    end_date = str(parameters["end_date"])
    log.info(f"Start the process of create upsell target for {date}")
    
    # Check if target can be created (date + max window < previous sunday)
    target_parameters = parameters["targets"][table_name]
    max_window = max([target_parameters[x] for x in target_parameters.keys() if x.endswith("window")])
    upper_bound = (pd.to_datetime(date) + timedelta(days=max_window)).strftime("%Y%m%d")
    previous_sunday = dt.today() - timedelta(days=dt.today().weekday() + 1)
    
    if pd.to_datetime(upper_bound, format="%Y%m%d") > previous_sunday:
        log.info(f"Cannot create upsell target for {date}: Not enough future information")
        return None

    # Compare with what is already processed
    path = f"{parameters['paths']['target_path']}{table_name}/"
    os.makedirs(path, exist_ok=True)
    processed_dates = os.listdir(path)
    match = [file for file in processed_dates if str(date) in file]
    if len(match) > 0 and overwrite is False:
        # If table is found, read parquet:
        log.info(f"Reading {match[0]} table")
        df_final = pd.read_parquet(path + match[0], engine="pyarrow")

    else:
        product_rank = parameters["targets"]["target_upsell"]["upsell_products_rank"]
        products_allowed_to_move=parameters["targets"]["target_upsell"]["products_allowed_to_move"]
        product_tecnology=parameters["targets"]["target_upsell"]["product_and_tecnology"]
        product_tecnology = {value : key for (key, value) in product_tecnology.items()}

        start_date = date
        end_date = (pd.to_datetime(date) + timedelta(days=parameters["targets"]["target_upsell"]["calculation_window"])).strftime("%Y%m%d")
        cancel_date = (pd.to_datetime(date) + timedelta(days=parameters["targets"]["target_upsell"]["activation_window"])).strftime("%Y%m%d")

        # Get EoP active clients from previous period to exclude new clients
        period_to_load = get_previous_month(start_date)
        df_clientes = eop.filter_by(condition=f"PRC_TIPO_ID = 3 AND PRC_CODIGO  IN {tuple(products_allowed_to_move)}",
                                     #base of customers that can made an upgrade
                                     date=period_to_load)

        # Get the user tecnology
        df_clientes["tecno_eop"]=df_clientes["PRC_CODIGO"].map(product_tecnology)
        df_clientes["tecno_eop"]=[y.split(" ")[1] for x,y in enumerate(df_clientes["tecno_eop"])]

        # Get data for target creation
        moves=tuple([ value[0] for (key,value) in product_rank.items()])
        df_upgrades = upgrades_basicos.filter_by(condition=f"EVENTO_ID = 108 AND PRODUCTO_ID IN {moves}",
                                                 date=[start_date, end_date],
                                                 target=True)
        #Tecnology of the basic product.
        df_upgrades["tecno_up"]=[y.split(" ")[1] for x,y in enumerate(df_upgrades["PRODUCTO"])]

        df_cancelations = upgrades_basicos.filter_by(date=[start_date,
                                                           cancel_date],
                                                     target=True)

        df_clientes_upgrades = pd.merge(
            df_clientes[["CUSTOMER_ID", "PRC_CODIGO","tecno_eop"]],
            df_upgrades[["CUSTOMER_ID", "PRODUCTO_ID", "FECHA","tecno_up"]],
            on=["CUSTOMER_ID"],
            how="inner",
            validate="1:m")
        del df_upgrades;
        gc.collect()

        df_clientes_upgrades.sort_values(["CUSTOMER_ID", "PRC_CODIGO", "FECHA"], ascending=[False, False, True],
                                         inplace=True)
        df_clientes_upgrades.drop_duplicates(subset=["CUSTOMER_ID", "PRC_CODIGO"], keep="last", inplace=True)

        df_product_rank = pd.DataFrame(product_rank.items(), columns=["PRODUCTO_RANK_INI", "PRC_CODIGO"])
        df_product_rank = df_product_rank.explode("PRC_CODIGO")


            # Rank initial product (PRC_CODIGO) from EOP table
        df_clientes_upgrades_ranked = pd.merge(df_clientes_upgrades,
                                               df_product_rank,
                                               on="PRC_CODIGO",
                                               how="left",
                                               validate="m:1")

        del df_clientes_upgrades
        gc.collect()

        # Rank last product (PRODUCTO_ID) from plan_evento table
        df_product_rank.rename(columns={"PRC_CODIGO": "PRODUCTO_ID",
                                        "PRODUCTO_RANK_INI": "PRODUCTO_RANK_END"}, inplace=True)
        df_clientes_upgrades_ranked = pd.merge(df_clientes_upgrades_ranked,
                                               df_product_rank,
                                               on="PRODUCTO_ID",
                                               how="left",
                                               validate="m:1")

        # Calculate target based on initial and end product plus tecnology
        df_clientes_upgrades_ranked["TARGET"] = np.where((df_clientes_upgrades_ranked["PRODUCTO_RANK_END"] > \
                                                         df_clientes_upgrades_ranked["PRODUCTO_RANK_INI"]) & ((df_clientes_upgrades_ranked["tecno_eop"] == \
                                                                                                               df_clientes_upgrades_ranked["tecno_up"])), 1, 0)

        # Keep only first move by CUSTOMER, PRODUCT
        df_cancelations.sort_values(["CUSTOMER_ID", "PRODUCTO_ID", "FECHA"], ascending=[False, False, True], inplace=True)
        df_cancelations.drop_duplicates(subset=["CUSTOMER_ID", "PRODUCTO_ID"], keep="last", inplace=True)

        # Merge with target df to check for activation period
        df_target = pd.merge(df_clientes_upgrades_ranked,
                             df_cancelations[["CUSTOMER_ID", "PRODUCTO_ID", "FECHA"]],
                             on=["CUSTOMER_ID", "PRODUCTO_ID"],
                             how="left",
                             validate="1:m")

        del df_clientes_upgrades_ranked, df_cancelations;
        gc.collect()

        # Compute time difference between events
        df_target["DATE_DIFF"] = (df_target["FECHA_y"] - df_target["FECHA_x"]) / np.timedelta64(1, "D")
        log.info(f" Number of events 108 ending as upgrades before product changes rule {df_target.TARGET.sum()}")

        df_target["TARGET"] = np.where((df_target["DATE_DIFF"] > 0) & \
                                       (df_target["DATE_DIFF"] <= parameters["targets"]["target_upsell"][
                                           "activation_window"]),
                                       0,
                                       df_target["TARGET"])
        df_target = drop_extra_rename_remaining(df_target)
        log.info(f" Number of events 108 ending as upgrades after product changes rule {df_target.TARGET.sum()}")

        # Merge back to EOP
        df_final = pd.merge(df_clientes[["CUSTOMER_ID", "PRC_CODIGO"]],
                            df_target[["CUSTOMER_ID", "TARGET", "FECHA", "PRODUCTO_ID"]],
                            on="CUSTOMER_ID",
                            how="left",
                            validate="1:1")
        
        target=df_final.loc[df_final.CUSTOMER_ID.isin(cliente_activo.CUSTOMER_ID.unique())]
        
        del df_target, df_final;
        gc.collect()

        target["TARGET"].fillna(0, inplace=True)
        target["TARGET"] =  target["TARGET"].astype(np.int32)
        target["DATE_EXP"] = period_to_load
        target["DATE_CALC"] = date
        target.rename({"FECHA": "FECHA_TARGET"}, inplace=True)
        
        if write_to_parquet:
            file = f"{parameters['paths']['target_path']}{table_name}/{table_name}_{date}.parquet"
            target.to_parquet(file, engine="pyarrow")

        # Return
        log.info(
            f"""Exporting target for period {start_date} and rate {
            np.round(100 * target[target['TARGET'] == 1]['CUSTOMER_ID'].nunique() / target['CUSTOMER_ID'].nunique(), 2)
            }%""")

    return target
    
    