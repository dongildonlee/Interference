from itertools import product
import numpy as np
import os
import pandas as pd
import pickle
import sys
sys.path.append('../')
from packages import stats, actv_analysis

class Unit:
    def __init__(self, unit):
        self.id = unit
        self.network = None
        self.relu = None
        self.epoch = None
        self.anova2 = None
        self.anova2_failed = False
        self.anova2_numbers = None
        self.anova2_sizes = None
        self.anova2_ns_int = None
        self.selectivity_number = None
        self.selectivity_size = None
        self.activity_raw = None
        self.activity_norm = None
        self.no_response = False
        self.no_response_subset = False
        self.response_rate = None
        self.response_rate_subset = None
        self.PN = None
        self.PS = None
        self.num_monotonicity = None
        self.size_monotonicity = None 

    def add_network_info(self,epoch,network,relu):
        self.epoch = epoch
        self.network = network
        self.relu = relu

    def add_anova2(self, actv_2D, numbers, sizes, inst, parallel=0):
        #result = stats.anova2_unit(actv_2D, self, numbers, sizes, inst, parallel)
        if parallel:
            return stats.anova2_unit(actv_2D, self, numbers, sizes, inst, parallel)
        else:
            stats.anova2_unit(actv_2D, self, numbers, sizes, inst, parallel)
        
    def to_dict(self):
        # Return the instance's data as a dictionary
        return vars(self)

        
    def add_activity(self, actv):
        self.activity_raw = actv[self.id,:,:]

    def add_selectivity(self):
        pass

    @classmethod
    def allowed_attributes(cls):
        """
        Returns a list of allowed attribute names for this class.
        """
        # Create an "empty" instance of this class and get its attributes
        empty_unit = cls(None)
        return [attr for attr in vars(empty_unit)]
    
    def clean_attributes(self):
        """
        Removes any attributes from this instance that are not allowed.
        """
        allowed_attributes = self.allowed_attributes()
        all_attributes = list(vars(self).keys())  # create a copy of keys list
        for attr in all_attributes:
            if attr not in allowed_attributes:
                delattr(self, attr)


    def is_complete(self):
        """
        Checks if a Unit object is complete by checking all its attributes.

        Returns:
        True if the Unit object is complete; otherwise, False.
        """
        attributes = ['id', 'network', 'relu', 'epoch', 'anova2', 'anova2_numbers', 'anova2_sizes', 
                      'selectivity_number', 'selectivity_size', 'activity_raw', 'activity_norm', 
                      'no_response', 'no_response_subset']

        return all(hasattr(self, attr) for attr in attributes)
    
    @classmethod
    def missing_attributes(cls, unit):
        """
        Checks if a Unit object has all necessary attributes.

        Parameters:
        - obj: A Unit object.

        Returns:
        - A list of missing attribute names, or an empty list if the object has all attributes.
        """
        # Get the list of all attribute names from an "empty" Unit object
        empty_unit = cls(None)
        attributes = [attr for attr in vars(empty_unit)]

        missing_attrs = [attr for attr in attributes if not hasattr(unit, attr)]

        return missing_attrs
    

    def fill_missing_attributes(self,net,relu,epoch,numbers=range(2,21,2), sizes = np.arange(4,14), min_sz_idx=3, max_sz_idx=9):
        missing_attrs = self.__class__.missing_attributes(self)
        if not missing_attrs:
            return

        actv_net_needed = any(attr in missing_attrs for attr in ['no_response', 'no_response_subset', 'PN', 'PS'])

        if actv_net_needed:
            actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)
            
        empty_unit = self.__class__(None)  # create an empty instance of the class
        default_values = vars(empty_unit)  # get default attribute values from the empty instance
        
        for attr in missing_attrs:
            if attr == 'no_response' and actv_net_needed:
                self.no_response = np.all(actv_net==0)
            elif attr == 'no_response_subset' and actv_net_needed:
                self.no_response_subset = np.all(self.find_no_response_sub(actv_net)==0)
            elif attr == 'PN' and actv_net_needed:
                self.PN = actv_analysis.get_PNs(actv_net, numbers, sizes, min_sz_idx, max_sz_idx)[1]
            elif attr == 'PS' and actv_net_needed:
                self.PS = actv_analysis.get_PSs(actv_net, numbers, sizes, min_sz_idx, max_sz_idx)[1]
            else:
                if attr in default_values:
                    print("adding attr..")
                    setattr(self, attr, default_values[attr])
                else:
                    print(f"Unknown attribute: {attr}")



    def find_no_response_sub(self, actv):
        numbers=np.arange(2,21,2)
        min_sz_idx=3
        max_sz_idx=9
        take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
        actv_szAtoB = actv[:,take,:]
        return actv_szAtoB

    def find_PN(self, avg_actv_nxs):
        """
        Finds and returns the Prefered Number (PN) for the unit.

        The PN is defined as the index along the first axis of 'avg_actv_nxs' that has the maximum value.

        Parameters:
        - avg_actv_nxs: A 3D numpy array. The first axis is units, the second is numbers, and the third is sizes.

        Returns:
        - A pandas DataFrame representing the PN for this unit. The row index is the unit id (self.id), 
        and the column names are sizes. The DataFrame contains the PN for each size.
        """
        
        # Define the sizes
        numbers = np.arange(2,21,2)
        sizes = np.arange(4,14)

        # For this unit (self.id), find the indices of the maximum values along the first axis of 'avg_actv_nxs'
        result = numbers[np.argmax(avg_actv_nxs[self.id,:,:], axis=0)]

        # Create a DataFrame from 'result' (reshaped to a 2D array), with row index as 'self.id' and column names as 'sizes'
        df_PN = pd.DataFrame(result.reshape(1, -1), index=[self.id], columns=sizes)

        return df_PN
    
    def find_PS(self, avg_actv_nxs, min_sz_idx=3, max_sz_idx=9):
        """
        Finds and returns the Prefered Size (PS) for the unit.

        The PN is defined as the index along the first axis of 'avg_actv_nxs' that has the maximum value.

        Parameters:
        - avg_actv_nxs: A 3D numpy array. The first axis is units, the second is numbers, and the third is sizes.

        Returns:
        - A pandas DataFrame representing the PS for this unit. The row index is the unit id (self.id), 
        and the column names are numbers. The DataFrame contains the PN for each size.
        """
        
        # Define the sizes
        numbers = np.arange(2,21,2)
        sizes = np.arange(4,14)

        # For this unit (self.id), find the indices of the maximum values along the first axis of 'avg_actv_nxs'
        result = sizes[np.argmax(avg_actv_nxs[self.id,:,:], axis=1)]
        result_subset = sizes[np.argmax(avg_actv_nxs[self.id,:,min_sz_idx:max_sz_idx+1], axis=1)]

        # Create a DataFrame from 'result' (reshaped to a 2D array), with row index as 'self.id' and column names as 'sizes'
        df_PS = pd.DataFrame(result.reshape(1, -1), index=[self.id], columns=numbers)
        df_PS_subset = pd.DataFrame(result_subset.reshape(1, -1), index=[self.id], columns=numbers)

        return df_PS, df_PS_subset


class Anova2ExistsError(Exception):
    """Raised when ANOVA2 already exists."""
    pass


def update_incomplete_units(net, relu, epoch, min_sz_idx=3, max_sz_idx=9, numbers=range(2,21,2)):
    print(f'network{net}_Relu{relu}_epoch{epoch}')
    pickle_filename = f'network{net}_Relu{relu}_epoch{epoch}.pkl'
    take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))

    if os.path.exists(pickle_filename):
        print("Loading pkl file..")
        with open(pickle_filename, 'rb') as f:
            units = pickle.load(f)
        
        incomplete_units = [unit for unit in units if not unit.is_complete()]
        if incomplete_units:
            print(f"Found {len(incomplete_units)} incomplete units.")
            actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)

            for i, unit in enumerate(incomplete_units):
                #print(f"Updating unit {i}")
                updated_unit = Unit(unit.id)
                updated_unit.__dict__ = unit.__dict__.copy()

                if not hasattr(unit, 'no_response_subset'):
                    updated_unit.no_response_subset = False

                actv_szAtoB = actv_net[unit.id, take, :]
                if np.all(actv_szAtoB == 0):
                    updated_unit.no_response_subset = True

                units[units.index(unit)] = updated_unit
                #print(f"Unit {i} updated.")
        else:
            print("All units have all attributes!")
    else:
        print("Pickle file does not exist, create new units..")
        actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)
        units = [Unit(i) for i in range(actv_net.shape[0])]
        for unit in units:
            unit.add_network_info(epoch, net, relu)
            unit.no_response = np.all(actv_net[unit.id, :, :] == 0)
            actv_szAtoB = actv_net[unit.id, take, :]
            unit.no_response_subset = np.all(actv_szAtoB == 0)

    with open(pickle_filename, 'wb') as f:
        #print(f"Saving {len(units)} units to {pickle_filename}...")
        pickle.dump(units, f)
        #print(f"Saved {len(units)} units to {pickle_filename}")

    return units
