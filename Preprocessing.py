import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.read_data()
        self.df_cleaned = None
        self.df_numeric = None

    def read_data(self):
        """Read data from the source file."""
        return pd.read_csv(self.file_path)

    def clean_data(self):
        """Preprocess the data to handle missing values and drop low correlation columns."""
        # Check for any NaN values and replace empty strings with NaN
        if self.df.isnull().sum().sum() != 0:
            self.df.replace('', np.nan, inplace=True)
            self.df.fillna(value=np.nan, inplace=True)
        
        # Drop columns with low correlation to the target
        self.df = self.df.drop(columns=['region', 'locality', 'latitude', 'longitude',
                                        'cadastral_income', 'subproperty_type', 'fl_open_fire',
                                        'construction_year','primary_energy_consumption_sqm',  'fl_double_glazing', 'id',
                                        'fl_floodzone', 'equipped_kitchen', 'fl_furnished',
                                        'fl_garden', 'fl_terrace', 'fl_swimming_pool'])
        
        # Remove rows with specific missing categorical values
        self.df_cleaned = self.df[~self.df['heating_type'].isin(['MISSING'])]
        self.df_cleaned = self.df_cleaned[~self.df_cleaned['state_building'].isin(['MISSING'])]
        self.df_cleaned = self.df_cleaned[~self.df_cleaned['epc'].isin(['MISSING'])]

    def preprocess_numeric_features(self):
        """Handle numeric features, including missing value imputation."""
        numeric_df = self.df_cleaned.select_dtypes(include='number')
        numeric_df.fillna({'surface_land_sqm': 0}, inplace=True)
        imputer = SimpleImputer(strategy='mean')
        self.df_numeric = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

    def preprocess_categorical_features(self):
        """Handle categorical features using one-hot and ordinal encoding."""
        # One-hot encode the 'property_type' column
        property_type = self.df_cleaned[['property_type']]
        enc = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
        property_type_encoded = enc.fit_transform(property_type)
        # Ensure matching indexes before joining
        property_type_encoded.index = self.df_numeric.index
        self.df_numeric = self.df_numeric.join(property_type_encoded)
        self.df_numeric.rename(columns={'property_type_HOUSE': 'property_type'}, inplace=True)

        # Ordinal encode the 'state_building' column
        building_state = self.df_cleaned[['state_building']]
        building_state_hierarchy = [
            'TO_RESTORE', 
            'TO_BE_DONE_UP', 
            'TO_RENOVATE', 
            'JUST_RENOVATED',
            'GOOD',
            'AS_NEW'
        ]
        encoder = OrdinalEncoder(categories=[building_state_hierarchy])
        building_state_encoded = encoder.fit_transform(building_state)

        building_state_encoded_df = pd.DataFrame(
            building_state_encoded, columns=['state_building'], index=self.df_numeric.index
        )
        building_state_encoded_df.index = self.df_numeric.index
        self.df_numeric = self.df_numeric.join(building_state_encoded_df)

        # Map the 'heating_type' column to an ordinal scale
        energy_order = {
            'CARBON': 0, 'WOOD': 1, 'PELLET': 2, 'FUELOIL': 3,
            'GAS': 4, 'ELECTRIC': 5, 'SOLAR': 6
        }
        heat_type_encoded = self.df_cleaned['heating_type'].map(energy_order)
        heat_type_encoded.index = self.df_numeric.index
        self.df_numeric = self.df_numeric.join(heat_type_encoded.rename("heating_type"))
        
        # Ordinal encode the 'province' column
        province = self.df_cleaned[["province"]]
        enc = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
        province_encoded = enc.fit_transform(province)
        # Make sure indexes match before joining to avoid NaN when joining
        province_encoded.index =  self.df_numeric.index 
        self.df_numeric = self.df_numeric.join(province_encoded)

        # Ordinal encode the 'epc' column
        epc = self.df_cleaned[['epc']]
        epc_hierarchy = [
            'G', 
            'F', 
            'E', 
            'D',
            'C',
            'B',
            'A',
            'A+',
            'A++',
        ]
        encoder = OrdinalEncoder(categories=[epc_hierarchy])
        epc_encoded = encoder.fit_transform(epc)

        epc_encoded_df = pd.DataFrame(
            epc_encoded, columns=['epc'], index=self.df_numeric.index
        )
        epc_encoded_df.index =  self.df_numeric.index 
        self.df_numeric = self.df_numeric.join(epc_encoded_df)

    def preprocess(self):
        """Execute all steps and return the final processed DataFrame."""
        self.clean_data()
        self.preprocess_numeric_features()
        self.preprocess_categorical_features()
        return self.df_numeric
