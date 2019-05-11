import pandas as pd
import numpy as np


def main():

    predictions_real= np.loadtxt("Predictions.txt" )

    print("prediction loaded \n")
    path="C://Users//david\Desktop//VU amsterdam//Data mining"
    new_test_set = pd.read_csv(path+"/New_test_set.csv")
    print(new_test_set)
    new_test_set = new_test_set.drop(["Unnamed: 0","random_bool"], axis=1)

    predictions_df=pd.DataFrame(predictions_real)
    #submiss_test_set= predictions_df
    submiss_test_set=pd.DataFrame(new_test_set["srch_id"])
    submiss_test_set.columns=["srch_id"]
    submiss_test_set["ranking"] = predictions_df
    submiss_test_set["prop_id"]=new_test_set["prop_id"]
    print(submiss_test_set)

    test_sub1 = submiss_test_set.groupby(by="srch_id")
    i=0
    submission_results=pd.DataFrame()
    print("Before loop")
    for key, item in test_sub1:
        
        #print(test_sub1.get_group(key).sort_values(by="ranking",ascending=False), "\n\n")
        group= test_sub1.get_group(key).sort_values(by="ranking",ascending=False)
        #print(group)
        submission_results= submission_results.append(group, ignore_index=True)
        if i == 10000:
            i=0
            print(key)
        i+=1

    submission_results=submission_results.drop("ranking", axis=1)
    submission_results.to_csv(path+"/submission_test_set.csv")


if __name__ == '__main__':
    main()
