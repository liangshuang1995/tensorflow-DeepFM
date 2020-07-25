
# set the path-to-files
TRAIN_FILE = "../example/data/train.csv"
TEST_FILE = "../example/data/test.csv"

SUB_DIR = "../example/output"

NUM_CK_POINTS = 10

# 每5个step就保存一次checkpoint
CHECKPOINT_EVERY = 5


NUM_SPLITS = 3
RANDOM_SEED = 2019

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    # "workclass", "education", "marital_status",
    # "occupation", "relationship", "race", "sex",
    # "native_country"
]

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",

    # feature engineering
    "missing_feat", "ps_car_13_x_ps_reg_03"
]

IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]


FEATURE_COLS = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "target"
]
