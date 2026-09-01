import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
    drop_columns
)
from us_visa.entity.estimator import TargetValueMapping


class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:

            print("\n" + "=" * 80)
            print("INITIALIZING DATA TRANSFORMATION")
            print("=" * 80)

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            print("[DEBUG] Data ingestion artifact:")
            print(f"        Train file: {self.data_ingestion_artifact.trained_file_path}")
            print(f"        Test file : {self.data_ingestion_artifact.test_file_path}")

            print("\n[DEBUG] Reading schema file:")
            print(f"        Schema path: {SCHEMA_FILE_PATH}")

            self._schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

            print("[DEBUG] Schema loaded successfully")
            print(f"[DEBUG] Schema keys: {self._schema_config.keys()}")

        except Exception as e:
            print("\n[ERROR] Error during DataTransformation initialization")
            print(f"[ERROR] Exception type    : {type(e).__name__}")
            print(f"[ERROR] Exception message : {str(e)}")

            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:

            print("\n" + "-" * 80)
            print("READING DATA")
            print("-" * 80)

            print(f"[DEBUG] File path: {file_path}")

            df = pd.read_csv(file_path)

            print("[DEBUG] File read successfully")
            print(f"[DEBUG] Shape: {df.shape}")
            print(f"[DEBUG] Columns: {list(df.columns)}")

            print("\n[DEBUG] First 5 rows:")
            print(df.head())

            print("\n[DEBUG] Data types:")
            print(df.dtypes)

            return df

        except Exception as e:

            print("\n[ERROR] Error while reading data")
            print(f"[ERROR] File: {file_path}")
            print(f"[ERROR] Exception type    : {type(e).__name__}")
            print(f"[ERROR] Exception message : {str(e)}")

            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:

        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        print("\n" + "=" * 80)
        print("CREATING DATA TRANSFORMER")
        print("=" * 80)

        try:

            print("[DEBUG] Initializing transformers")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")
            ordinal_encoder = OrdinalEncoder()

            print("[DEBUG] Transformers initialized:")
            print(f"        Numeric transformer : {numeric_transformer}")
            print(f"        OneHot transformer  : {oh_transformer}")
            print(f"        Ordinal transformer: {ordinal_encoder}")

            self.oh_columns = self._schema_config['oh_columns']
            self.or_columns = self._schema_config['or_columns']
            self.transform_columns = self._schema_config['transform_columns']
            self.num_features = self._schema_config['num_features']

            print("\n[DEBUG] Columns from schema:")
            print(f"        OneHot columns       : {self.oh_columns}")
            print(f"        Ordinal columns      : {self.or_columns}")
            print(f"        Transform columns    : {self.transform_columns}")
            print(f"        Numerical features   : {self.num_features}")

            print("\n[DEBUG] Initializing PowerTransformer")

            transform_pipe = Pipeline(
                steps=[
                    (
                        'transformer',
                        PowerTransformer(method='yeo-johnson')
                    )
                ]
            )

            print("[DEBUG] PowerTransformer initialized")

            print("\n[DEBUG] Creating ColumnTransformer")

            preprocessor = ColumnTransformer(
                [
                    (
                        "OneHotEncoder",
                        oh_transformer,
                        self.oh_columns
                    ),
                    (
                        "Ordinal_Encoder",
                        ordinal_encoder,
                        self.or_columns
                    ),
                    (
                        "Transformer",
                        transform_pipe,
                        self.transform_columns
                    ),
                    (
                        "StandardScaler",
                        numeric_transformer,
                        self.num_features
                    )
                ]
            )

            print("[DEBUG] ColumnTransformer created successfully")

            print("\n[DEBUG] Transformer configuration:")
            print(preprocessor)

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return preprocessor

        except Exception as e:

            print("\n[ERROR] Error creating data transformer")
            print(f"[ERROR] Exception type    : {type(e).__name__}")
            print(f"[ERROR] Exception message : {str(e)}")

            raise USvisaException(e, sys) from e

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:

        print("\n" + "#" * 80)
        print("STARTING DATA TRANSFORMATION PIPELINE")
        print("#" * 80)

        try:

            # ------------------------------------------------------------------
            # VALIDATION STATUS
            # ------------------------------------------------------------------

            print("\n[STEP 1] Checking data validation status")

            print(
                f"[DEBUG] Validation status: "
                f"{self.data_validation_artifact.validation_status}"
            )

            print(
                f"[DEBUG] Validation message: "
                f"{self.data_validation_artifact.message}"
            )

            if self.data_validation_artifact.validation_status:

                logging.info("Starting data transformation")

                print("\n[STEP 2] Creating preprocessor")

                preprocessor = self.get_data_transformer_object()

                print("[DEBUG] Preprocessor successfully created")

                # ------------------------------------------------------------------
                # READ TRAIN DATA
                # ------------------------------------------------------------------

                print("\n[STEP 3] Reading training data")

                train_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                )

                print(
                    f"[DEBUG] Training dataframe shape: {train_df.shape}"
                )

                # ------------------------------------------------------------------
                # READ TEST DATA
                # ------------------------------------------------------------------

                print("\n[STEP 4] Reading testing data")

                test_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                )

                print(
                    f"[DEBUG] Testing dataframe shape: {test_df.shape}"
                )

                # ------------------------------------------------------------------
                # CHECK TARGET COLUMN
                # ------------------------------------------------------------------

                print("\n[STEP 5] Checking target column")

                print(f"[DEBUG] TARGET_COLUMN = {TARGET_COLUMN}")

                print(
                    f"[DEBUG] Target exists in train: "
                    f"{TARGET_COLUMN in train_df.columns}"
                )

                print(
                    f"[DEBUG] Target exists in test : "
                    f"{TARGET_COLUMN in test_df.columns}"
                )

                if TARGET_COLUMN not in train_df.columns:
                    raise Exception(
                        f"Target column '{TARGET_COLUMN}' "
                        f"does not exist in training dataframe"
                    )

                if TARGET_COLUMN not in test_df.columns:
                    raise Exception(
                        f"Target column '{TARGET_COLUMN}' "
                        f"does not exist in testing dataframe"
                    )

                # ------------------------------------------------------------------
                # SPLIT TRAIN FEATURES / TARGET
                # ------------------------------------------------------------------

                print("\n[STEP 6] Splitting training features and target")

                input_feature_train_df = train_df.drop(
                    columns=[TARGET_COLUMN]
                )

                target_feature_train_df = train_df[TARGET_COLUMN]

                print(
                    f"[DEBUG] Train feature shape: "
                    f"{input_feature_train_df.shape}"
                )

                print(
                    f"[DEBUG] Train target shape: "
                    f"{target_feature_train_df.shape}"
                )

                print("\n[DEBUG] Train target values BEFORE mapping:")

                print(
                    target_feature_train_df.value_counts(
                        dropna=False
                    )
                )

                print("\n[DEBUG] Unique train target values:")
                print(
                    target_feature_train_df.unique()
                )

                # ------------------------------------------------------------------
                # COMPANY AGE - TRAIN
                # ------------------------------------------------------------------

                print("\n[STEP 7] Creating company_age for training data")

                print(
                    f"[DEBUG] CURRENT_YEAR = {CURRENT_YEAR}"
                )

                print(
                    f"[DEBUG] yr_of_estab exists: "
                    f"{'yr_of_estab' in input_feature_train_df.columns}"
                )

                if 'yr_of_estab' not in input_feature_train_df.columns:
                    raise Exception(
                        "'yr_of_estab' column is missing from training data"
                    )

                print("\n[DEBUG] yr_of_estab sample:")
                print(
                    input_feature_train_df['yr_of_estab'].head()
                )

                input_feature_train_df['company_age'] = (
                    CURRENT_YEAR -
                    input_feature_train_df['yr_of_estab']
                )

                print(
                    "[DEBUG] company_age successfully created"
                )

                print("\n[DEBUG] company_age sample:")
                print(
                    input_feature_train_df['company_age'].head()
                )

                print("\n[DEBUG] company_age statistics:")
                print(
                    input_feature_train_df['company_age'].describe()
                )

                # ------------------------------------------------------------------
                # DROP COLUMNS - TRAIN
                # ------------------------------------------------------------------

                drop_cols = self._schema_config['drop_columns']

                print("\n[STEP 8] Dropping training columns")

                print(f"[DEBUG] drop_columns from schema: {drop_cols}")

                print(
                    f"[DEBUG] Train columns BEFORE dropping:"
                )
                print(
                    list(input_feature_train_df.columns)
                )

                input_feature_train_df = drop_columns(
                    df=input_feature_train_df,
                    cols=drop_cols
                )

                print(
                    f"[DEBUG] Train columns AFTER dropping:"
                )
                print(
                    list(input_feature_train_df.columns)
                )

                print(
                    f"[DEBUG] Train feature shape AFTER dropping: "
                    f"{input_feature_train_df.shape}"
                )

                # ------------------------------------------------------------------
                # MAP TRAIN TARGET
                # ------------------------------------------------------------------

                print("\n[STEP 9] Mapping training target values")

                target_mapping = TargetValueMapping()._asdict()

                print(
                    f"[DEBUG] Target mapping: {target_mapping}"
                )

                target_feature_train_df = target_feature_train_df.apply(
                    lambda x: target_mapping["mapping"].get(x, "Not Found")
                )

                print(
                    "\n[DEBUG] Train target values AFTER mapping:"
                )

                print(
                    target_feature_train_df.value_counts(
                        dropna=False
                    )
                )

                print("\n[DEBUG] Unique mapped train targets:")
                print(
                    target_feature_train_df.unique()
                )

                # Check for unmapped values
                if target_feature_train_df.isna().any():

                    print(
                        "\n[ERROR] NaN values found in mapped "
                        "training target!"
                    )

                    print(
                        train_df[TARGET_COLUMN]
                        [
                            target_feature_train_df.isna()
                        ]
                        .unique()
                    )

                    raise Exception(
                        "Target mapping produced NaN values in training target"
                    )

                # ------------------------------------------------------------------
                # SPLIT TEST FEATURES / TARGET
                # ------------------------------------------------------------------

                print("\n[STEP 10] Splitting testing features and target")

                input_feature_test_df = test_df.drop(
                    columns=[TARGET_COLUMN]
                )

                target_feature_test_df = test_df[TARGET_COLUMN]

                print(
                    f"[DEBUG] Test feature shape: "
                    f"{input_feature_test_df.shape}"
                )

                print(
                    f"[DEBUG] Test target shape: "
                    f"{target_feature_test_df.shape}"
                )

                print("\n[DEBUG] Test target values BEFORE mapping:")

                print(
                    target_feature_test_df.value_counts(
                        dropna=False
                    )
                )

                # ------------------------------------------------------------------
                # COMPANY AGE - TEST
                # ------------------------------------------------------------------

                print("\n[STEP 11] Creating company_age for test data")

                print(
                    f"[DEBUG] yr_of_estab exists in test: "
                    f"{'yr_of_estab' in input_feature_test_df.columns}"
                )

                if 'yr_of_estab' not in input_feature_test_df.columns:
                    raise Exception(
                        "'yr_of_estab' column is missing from test data"
                    )

                input_feature_test_df['company_age'] = (
                    CURRENT_YEAR -
                    input_feature_test_df['yr_of_estab']
                )

                print(
                    "[DEBUG] company_age successfully created for test"
                )

                print("\n[DEBUG] Test company_age sample:")
                print(
                    input_feature_test_df['company_age'].head()
                )

                # ------------------------------------------------------------------
                # DROP COLUMNS - TEST
                # ------------------------------------------------------------------

                print("\n[STEP 12] Dropping testing columns")

                print(
                    "[DEBUG] Test columns BEFORE dropping:"
                )

                print(
                    list(input_feature_test_df.columns)
                )

                input_feature_test_df = drop_columns(
                    df=input_feature_test_df,
                    cols=drop_cols
                )

                print(
                    "[DEBUG] Test columns AFTER dropping:"
                )

                print(
                    list(input_feature_test_df.columns)
                )

                print(
                    f"[DEBUG] Test feature shape AFTER dropping: "
                    f"{input_feature_test_df.shape}"
                )

                # ------------------------------------------------------------------
                # MAP TEST TARGET
                # ------------------------------------------------------------------

                print("\n[STEP 13] Mapping testing target values")

                target_feature_test_df = target_feature_test_df.apply(
                    lambda x: target_mapping["mapping"].get(x, "Not Found")
                )

                print(
                    "\n[DEBUG] Test target values AFTER mapping:"
                )

                print(
                    target_feature_test_df.value_counts(
                        dropna=False
                    )
                )

                print("\n[DEBUG] Unique mapped test targets:")
                print(
                    target_feature_test_df.unique()
                )

                if target_feature_test_df.isna().any():

                    print(
                        "\n[ERROR] NaN values found in mapped "
                        "testing target!"
                    )

                    print(
                        test_df[TARGET_COLUMN]
                        [
                            target_feature_test_df.isna()
                        ]
                        .unique()
                    )

                    raise Exception(
                        "Target mapping produced NaN values in testing target"
                    )

                # ------------------------------------------------------------------
                # VERIFY TRAIN / TEST COLUMNS MATCH
                # ------------------------------------------------------------------

                print("\n[STEP 14] Comparing train and test columns")

                train_columns = set(
                    input_feature_train_df.columns
                )

                test_columns = set(
                    input_feature_test_df.columns
                )

                print(
                    f"[DEBUG] Columns in train but NOT test: "
                    f"{train_columns - test_columns}"
                )

                print(
                    f"[DEBUG] Columns in test but NOT train: "
                    f"{test_columns - train_columns}"
                )

                if train_columns != test_columns:

                    raise Exception(
                        "Training and testing feature columns do not match"
                    )

                print(
                    "[DEBUG] Train and test columns match"
                )

                # ------------------------------------------------------------------
                # VERIFY SCHEMA COLUMNS
                # ------------------------------------------------------------------

                print("\n[STEP 15] Verifying schema columns")

                all_schema_columns = (
                    self.oh_columns
                    + self.or_columns
                    + self.transform_columns
                    + self.num_features
                )

                print(
                    f"[DEBUG] All transformer columns: "
                    f"{all_schema_columns}"
                )

                missing_schema_columns = [
                    col
                    for col in all_schema_columns
                    if col not in input_feature_train_df.columns
                ]

                print(
                    f"[DEBUG] Missing schema columns in train: "
                    f"{missing_schema_columns}"
                )

                if missing_schema_columns:

                    raise Exception(
                        f"These columns required by the preprocessor "
                        f"are missing: {missing_schema_columns}"
                    )

                # ------------------------------------------------------------------
                # PREPROCESSING
                # ------------------------------------------------------------------

                print("\n[STEP 16] Applying preprocessor to training data")

                print(
                    f"[DEBUG] Input train shape: "
                    f"{input_feature_train_df.shape}"
                )

                print(
                    "[DEBUG] Starting preprocessor.fit_transform()..."
                )

                input_feature_train_arr = (
                    preprocessor.fit_transform(
                        input_feature_train_df
                    )
                )

                print(
                    "[DEBUG] preprocessor.fit_transform() completed"
                )

                print(
                    f"[DEBUG] Transformed train shape: "
                    f"{input_feature_train_arr.shape}"
                )

                print(
                    f"[DEBUG] Transformed train type: "
                    f"{type(input_feature_train_arr)}"
                )

                # ------------------------------------------------------------------
                # TEST TRANSFORMATION
                # ------------------------------------------------------------------

                print("\n[STEP 17] Applying preprocessor to test data")

                print(
                    f"[DEBUG] Input test shape: "
                    f"{input_feature_test_df.shape}"
                )

                print(
                    "[DEBUG] Starting preprocessor.transform()..."
                )

                input_feature_test_arr = (
                    preprocessor.transform(
                        input_feature_test_df
                    )
                )

                print(
                    "[DEBUG] preprocessor.transform() completed"
                )

                print(
                    f"[DEBUG] Transformed test shape: "
                    f"{input_feature_test_arr.shape}"
                )

                print(
                    f"[DEBUG] Transformed test type: "
                    f"{type(input_feature_test_arr)}"
                )

                # ------------------------------------------------------------------
                # SMOTEENN - TRAIN
                # ------------------------------------------------------------------

                print("\n[STEP 18] Applying SMOTEENN to TRAINING data")

                print(
                    "[DEBUG] Training class distribution BEFORE SMOTEENN:"
                )

                print(
                    pd.Series(
                        target_feature_train_df
                    ).value_counts()
                )

                print(
                    f"[DEBUG] Train feature shape BEFORE SMOTEENN: "
                    f"{input_feature_train_arr.shape}"
                )

                print(
                    f"[DEBUG] Train target shape BEFORE SMOTEENN: "
                    f"{target_feature_train_df.shape}"
                )

                smt = SMOTEENN(
                    sampling_strategy="minority"
                )

                print(
                    "[DEBUG] SMOTEENN object created"
                )

                print(
                    "[DEBUG] Starting SMOTEENN.fit_resample() "
                    "on TRAINING data..."
                )

                input_feature_train_final, target_feature_train_final = (
                    smt.fit_resample(
                        input_feature_train_arr,
                        target_feature_train_df
                    )
                )

                print(
                    "[DEBUG] SMOTEENN training completed"
                )

                print(
                    f"[DEBUG] Train feature shape AFTER SMOTEENN: "
                    f"{input_feature_train_final.shape}"
                )

                print(
                    f"[DEBUG] Train target shape AFTER SMOTEENN: "
                    f"{target_feature_train_final.shape}"
                )

                print(
                    "\n[DEBUG] Training class distribution "
                    "AFTER SMOTEENN:"
                )

                print(
                    pd.Series(
                        target_feature_train_final
                    ).value_counts()
                )

                # ------------------------------------------------------------------
                # SMOTEENN - TEST
                # ------------------------------------------------------------------

                print("\n[STEP 19] Applying SMOTEENN to TESTING data")

                print(
                    "[DEBUG] Test class distribution BEFORE SMOTEENN:"
                )

                print(
                    pd.Series(
                        target_feature_test_df
                    ).value_counts()
                )

                print(
                    f"[DEBUG] Test feature shape BEFORE SMOTEENN: "
                    f"{input_feature_test_arr.shape}"
                )

                print(
                    f"[DEBUG] Test target shape BEFORE SMOTEENN: "
                    f"{target_feature_test_df.shape}"
                )

                print(
                    "[DEBUG] Starting SMOTEENN.fit_resample() "
                    "on TESTING data..."
                )

                input_feature_test_final, target_feature_test_final = (
                    smt.fit_resample(
                        input_feature_test_arr,
                        target_feature_test_df
                    )
                )

                print(
                    "[DEBUG] SMOTEENN testing completed"
                )

                print(
                    f"[DEBUG] Test feature shape AFTER SMOTEENN: "
                    f"{input_feature_test_final.shape}"
                )

                print(
                    f"[DEBUG] Test target shape AFTER SMOTEENN: "
                    f"{target_feature_test_final.shape}"
                )

                print(
                    "\n[DEBUG] Testing class distribution "
                    "AFTER SMOTEENN:"
                )

                print(
                    pd.Series(
                        target_feature_test_final
                    ).value_counts()
                )

                # ------------------------------------------------------------------
                # CREATE FINAL ARRAYS
                # ------------------------------------------------------------------

                print("\n[STEP 20] Creating final train/test arrays")

                train_arr = np.c_[
                    input_feature_train_final,
                    np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final,
                    np.array(target_feature_test_final)
                ]

                print(
                    f"[DEBUG] Final train array shape: "
                    f"{train_arr.shape}"
                )

                print(
                    f"[DEBUG] Final test array shape: "
                    f"{test_arr.shape}"
                )

                print(
                    f"[DEBUG] Final train array dtype: "
                    f"{train_arr.dtype}"
                )

                print(
                    f"[DEBUG] Final test array dtype: "
                    f"{test_arr.dtype}"
                )

                # ------------------------------------------------------------------
                # SAVE PREPROCESSOR
                # ------------------------------------------------------------------

                print("\n[STEP 21] Saving preprocessor")

                print(
                    f"[DEBUG] Preprocessor path: "
                    f"{self.data_transformation_config.transformed_object_file_path}"
                )

                save_object(
                    self.data_transformation_config.transformed_object_file_path,
                    preprocessor
                )

                print(
                    "[DEBUG] Preprocessor saved successfully"
                )

                # ------------------------------------------------------------------
                # SAVE TRAIN ARRAY
                # ------------------------------------------------------------------

                print("\n[STEP 22] Saving transformed training data")

                print(
                    f"[DEBUG] Train output path: "
                    f"{self.data_transformation_config.transformed_train_file_path}"
                )

                save_numpy_array_data(
                    self.data_transformation_config.transformed_train_file_path,
                    array=train_arr
                )

                print(
                    "[DEBUG] Transformed training data saved successfully"
                )

                # ------------------------------------------------------------------
                # SAVE TEST ARRAY
                # ------------------------------------------------------------------

                print("\n[STEP 23] Saving transformed testing data")

                print(
                    f"[DEBUG] Test output path: "
                    f"{self.data_transformation_config.transformed_test_file_path}"
                )

                save_numpy_array_data(
                    self.data_transformation_config.transformed_test_file_path,
                    array=test_arr
                )

                print(
                    "[DEBUG] Transformed testing data saved successfully"
                )

                # ------------------------------------------------------------------
                # CREATE ARTIFACT
                # ------------------------------------------------------------------

                print("\n[STEP 24] Creating DataTransformationArtifact")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=(
                        self.data_transformation_config
                        .transformed_object_file_path
                    ),
                    transformed_train_file_path=(
                        self.data_transformation_config
                        .transformed_train_file_path
                    ),
                    transformed_test_file_path=(
                        self.data_transformation_config
                        .transformed_test_file_path
                    )
                )

                print(
                    "[DEBUG] DataTransformationArtifact created successfully"
                )

                print("\n" + "#" * 80)
                print("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
                print("#" * 80)

                print("\n[FINAL OUTPUT]")
                print(
                    f"Transformed object: "
                    f"{data_transformation_artifact.transformed_object_file_path}"
                )
                print(
                    f"Transformed train: "
                    f"{data_transformation_artifact.transformed_train_file_path}"
                )
                print(
                    f"Transformed test : "
                    f"{data_transformation_artifact.transformed_test_file_path}"
                )

                return data_transformation_artifact

            else:

                print("\n[ERROR] DATA VALIDATION FAILED")

                print(
                    f"[ERROR] Validation message: "
                    f"{self.data_validation_artifact.message}"
                )

                raise Exception(
                    self.data_validation_artifact.message
                )

        except Exception as e:

            print("\n" + "!" * 80)
            print("DATA TRANSFORMATION FAILED")
            print("!" * 80)

            print(
                f"[ERROR] Exception type    : {type(e).__name__}"
            )

            print(
                f"[ERROR] Exception message : {str(e)}"
            )

            print(
                f"[ERROR] Exception repr    : {repr(e)}"
            )

            logging.exception(
                "Exception occurred during data transformation"
            )

            raise USvisaException(e, sys) from e