#!/bin/bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import dump_svmlight_file

def features_addition(dataset, test):
    data=dataset
    data["starrating_diff"] = abs(data["visitor_hist_starrating"] - data["prop_starrating"])
    data["usd_diff"] = abs(np.log10(data["visitor_hist_adr_usd"]) - np.log10(data["price_usd"]))
    
    data = data.fillna(value = {"starrating_diff": 6, "usd_diff": 1.1})
    #difference between stars and review stars
    data['ad_vs_real'] = data['prop_starrating'] - data['prop_review_score']



    if test==True:
        path="C://Users//david\Desktop//VU amsterdam//Data mining"
        hotels_train = pd.read_csv(path+"//hotel_quality.csv")
        hotels_train.columns=["prop_id", "booked_perc", "clicked_perc"]
        data=data.join(hotels_train.set_index("prop_id"),how="left", on="prop_id")
        #filling Nan with mean
        data = data.fillna(value = {"booked_perc": 1.92, "clicked_perc": 2.74})

    else:
        # Hotel quality
        # times booked / times in the data
        # times clicked / times in the data
        hotel_quality = pd.DataFrame(dataset.prop_id.value_counts(dropna = False))
        #print(hotel_quality.head())
        hotel_quality = hotel_quality.join(pd.DataFrame(dataset.prop_id[dataset.booking_bool == 1].value_counts().astype(int)), rsuffix = "book")
        hotel_quality = hotel_quality.join(pd.DataFrame(dataset.prop_id[dataset.click_bool == 1].value_counts().astype(int)), rsuffix = "click")
        hotel_quality.columns = ["counts", "booked", "clicked"]

        hotel_quality["booked_percentage"] = hotel_quality.booked / hotel_quality.counts * 100
        hotel_quality["clicked_percentage"] = hotel_quality.clicked / hotel_quality.counts * 100

        data = data.join(hotel_quality.booked_percentage, on = "prop_id")
        data = data.join(hotel_quality.clicked_percentage, on = "prop_id")
        #filling Nan with mean
        data = data.fillna(value = {"booked_percentage": 1.92, "clicked_percentage": 2.74})
    
    # Average comp price
    data['avg_comp_rate'] = data[['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']].mean(axis=1)
    data = data.drop(['comp1_rate', "comp1_inv", "comp1_rate_percent_diff", 'comp2_rate', "comp2_inv", "comp2_rate_percent_diff", 'comp3_rate', "comp3_inv", "comp3_rate_percent_diff", 'comp4_rate', "comp4_inv", "comp4_rate_percent_diff", 'comp5_rate', "comp5_inv", "comp5_rate_percent_diff", 'comp6_rate', "comp6_inv", "comp6_rate_percent_diff", 'comp7_rate', "comp7_inv", "comp7_rate_percent_diff", 'comp8_rate', "comp8_inv", "comp8_rate_percent_diff"], axis = 1)
    data = data.fillna(value = {"avg_comp_rate": 0}) 
    
    #add month columns
    data["date_time"] = pd.to_datetime(data["date_time"])
    data["month"] = data["date_time"].dt.month
    data = data.drop("date_time", axis=1)

    #Compose features roomcount_bookwindow, adultcount_childrencount by F1_F2 = F1*max(Max(F2)) + F2
    data['roomcount_bookwindow'] = data['srch_room_count']*max(data['srch_booking_window']) + data['srch_booking_window']
    data['adultcount_childrencount'] = data['srch_adults_count']*max(data['srch_children_count']) + data['srch_children_count']

    #1 if hotel is in the same country
    data['bool_same_country'] = 0
    data.loc[(data['visitor_location_country_id'] == data['prop_country_id']), 'bool_same_country'] = 1
    #there are some infinite values in usd_diff
    data= data.replace([np.inf, -np.inf], np.nan)
    #train.columns[train.isna().any()].tolist()
    data=data.fillna(value={"usd_diff":1.1})

    return data

def fill_nan(data):
    #Drop "orig_destination_distance"
    #data.drop("orig_destination_distance", axis=1, inplace=True)
    #replace NaN for zero
    data["orig_destination_distance"]=data["orig_destination_distance"].replace(np.nan, data["orig_destination_distance"].quantile(0.75))
    data["prop_review_score"] = data["prop_review_score"].replace(np.nan, 0)
    data["prop_location_score2"] = data["prop_location_score2"].replace(np.nan, data["prop_location_score2"].median(axis = 0, skipna=True))
    
    #replace NaN for worst case scenario, in this case -326.567500 which is the minimum value for this feature
    data["srch_query_affinity_score"] = data["srch_query_affinity_score"].replace(np.nan, -326.567500)

    #fill with median
    #data=data.drop("gross_bookings_usd",axis=1)
    median_vhs=data["visitor_hist_starrating"].median(axis = 0, skipna=True)
    median_vha=data["visitor_hist_adr_usd"].median(axis = 0, skipna=True) 
    values = {'visitor_hist_starrating': median_vhs, 'visitor_hist_adr_usd': median_vha}
    data=data.fillna(value=values)

    #removing outliers
    indices_to_remove = data[data['price_usd'] > 50000].index.tolist()
    data = data.drop(indices_to_remove)

    return data

def balancing_dataset(data):
    df=data.groupby('srch_id')["booking_bool"].apply(lambda x: (x==1).sum()).reset_index(name='count')
    df_count=df.groupby("count")
    #ids' of booked and not booked searches
    df_booked=df_count.get_group(1).drop("count",axis=1)
    df_not_booked=df_count.get_group(0).drop("count",axis=1)
    #CHANGE this variable to  change size
    dimension_not_booked=30000

    #sampling  using the selected ids'
    df_booked = df_booked.sample(n=dimension_not_booked,random_state=22).reset_index()
    df_not_booked = df_not_booked.sample(n=dimension_not_booked, random_state=13).reset_index()
    #drop useless column
    df_not_booked=df_not_booked.drop("index", axis=1)
    df_booked=df_booked.drop("index", axis=1)

    #merge basedo on searches is with the original dataset
    all_booked=data.merge(df_booked, on="srch_id")
    all_not_booked=data.merge(df_not_booked, on="srch_id")
    #put together booked and not booked
    final_train=pd.concat([all_not_booked, all_booked]).reset_index()
    final_train=final_train.drop("index",axis=1)

    return final_train

def assign_score(x):
    if x["booking_bool"]==1:
        val=5
    elif x["click_bool"]==1:
        val=1
    else:
        val=0
    return val
    

def main():
    print("Reading the file...")
    path="C://Users//david\Desktop//VU amsterdam//Data mining"
    data = pd.read_csv(path+"//training_set_VU_DM.csv")
    print("original dataset: \n")
    print(data)
    #downsampling the dataset
    #after_reduction=balancing_dataset(data)
    # fill Nan
    #print("Before fill Nan: \n")
    print("Filling the Nan..")
    after_reduction=data
    #print(after_reduction)
    after_nan=fill_nan(after_reduction)
    #add new features
    print("Before add new features: \n")
    print(after_nan)
    print("Adding some features...")
    #CHANGE lines BELOW for training
    test= False
    new_data = features_addition(after_nan, test)
    print("Final dataset: \n")
    print(new_data)
    # add score column (only for train set!):
    #Adding Score columns: 5 for booked, 1 clicked and 0 the rest
    #print("Assigning the score...")
    new_data['score'] = new_data.apply(assign_score , axis=1)

    new_data=new_data.drop(["random_bool" ,"booking_bool", "click_bool","gross_bookings_usd","site_id"], axis=1)
    #new_data=new_data.drop(["random_bool" ,"site_id"], axis=1)
    #drop search id?
    #new_data = new_data.drop("srch_id", axis=1)
    #store resulting dataset
    print("Storing the preprocessed dataset...")
    new_data.to_csv(path+"/New_train_set_full.csv")



if __name__ == '__main__':
    main()
