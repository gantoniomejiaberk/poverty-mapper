import json 
import boto3
import botocore

def get_model_specs(split_name_list, num_classes_list, bin_method_list, pretrained_list, freeze_layers_list, epoch_list, learning_rate_list, gamma_list, step_size_list, batch_size_list, num_workers_list):
    """
    Take lists of model spec options and parameters and save 
    a .json file for each combination of parameters to s3.
    
    split_name_list:      list of string names of data splits from data_splits.json
    num_classes:          list of integer number of classes
    bin_method_list:      list of binning methods ("within", "across", "within_even", "across_even")
    pretrained_list:      list of string names of pytorch pretrained models
    freeze_layers_list:   list of options for whether or not to freeze layers ("yes, "no")
    epoch_list:           list of integer epoch values
    learning_rate_list:   list of float learning rate values  
    gamma_list:           list of integer gamma values
    step_size_list:       list of integer step sizes 
    batch_size_list:      list of integer batch sizes
    num_workers_list:     list of integer number of workers    
    """
    
    # Set boto3  client
    s3 = boto3.client("s3")
    
    bucket = "w210-poverty-mapper"
    
    # Set model data paths
    five_across_path = "modeling/data_splits/five_across/"
    two_02_across_path = "modeling/data_splits/two_02_across/"
    five_within_path = "modeling/data_splits/five_within/"
    two_02_within_path = "modeling/data_splits/two_02_within/"
    
    five_across_even_path = "modeling/data_splits/five_across_even/"
    two_02_across_even_path = "modeling/data_splits/two_02_across_even/"
    five_within_even_path = "modeling/data_splits/five_within_even/"
    two_02_within_even_path = "modeling/data_splits/two_02_within_even/"
    
    # Create dictionary for each combination
    for split_name in split_name_list:
        for num_classes in num_classes_list:
            for bin_method in bin_method_list:
                for pretrained in pretrained_list:
                    for option in freeze_layers_list:
                        for epochs in epoch_list:
                            for learning_rate in learning_rate_list:
                                for gamma in gamma_list:
                                    for step_size in step_size_list:
                                        for batch_size in batch_size_list:
                                            for num_workers in num_workers_list:

                                                # Get train/val/test paths
                                                if num_classes == 5 and bin_method == "across":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=five_across_path + split_name)['Contents']

                                                elif num_classes == 2 and bin_method == "across":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=two_02_across_path + split_name)['Contents']

                                                elif num_classes == 5 and bin_method == "within":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=five_within_path
                                                                                   + split_name)['Contents']

                                                elif num_classes == 2 and bin_method == "within":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=two_02_within_path + split_name)['Contents']
                                                
                                                elif num_classes == 5 and bin_method == "across_even":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=five_across_even_path + split_name)['Contents']

                                                elif num_classes == 2 and bin_method == "across_even":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=two_02_across_even_path + split_name)['Contents']

                                                elif num_classes == 5 and bin_method == "within_even":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=five_within_even_path
                                                                                   + split_name)['Contents']

                                                elif num_classes == 2 and bin_method == "within_even":
                                                        contents = s3.list_objects(Bucket=bucket,
                                                                                   Prefix=two_02_within_even_path + split_name)['Contents']
                                                
                                                else:
                                                    raise Exception("Not a valid bin class combination.")

                                                for f in contents:
                                                    if "train" in f["Key"]:
                                                        train = f["Key"]
                                                    if "val" in f["Key"]:
                                                        val = f["Key"]
                                                    if "test" in f["Key"]:
                                                        test = f["Key"]

                                                # Get filename
                                                spec_items = [split_name, str(num_classes), bin_method, pretrained, option, str(epochs), str(learning_rate), str(gamma), str(step_size),str(batch_size), str(num_workers)]
                                                filename = "_".join(spec_items)

                                                data_root = "s3://w210-poverty-mapper/"
                                                results_path = "s3://w210-poverty-mapper/modeling/results/" + filename + "_results.json"
                                                model_artifacts_path = "s3://w210-poverty-mapper/modeling/model_artifacts/"

                                                model_spec = {}
                                                model_spec["split_name"] = split_name
                                                model_spec["num_classes"] = num_classes
                                                model_spec["bin_method"] = bin_method
                                                model_spec["train"] = data_root + train
                                                model_spec["val"] = data_root + val
                                                model_spec["test"] = data_root + test
                                                model_spec["pretrained"] = pretrained
                                                model_spec["freeze_layers"] = option 
                                                model_spec["epochs"] = epochs 
                                                model_spec["learning_rate"] = learning_rate 
                                                model_spec["gamma"] = gamma 
                                                model_spec["step_size"] = step_size 
                                                model_spec["batch_size"] = batch_size
                                                model_spec["num_workers"] = num_workers
                                                model_spec["results_path"] = results_path
                                                model_spec["model_artifacts_path"] = model_artifacts_path

                                                # Write model spec to s3
                                                s3.put_object(
                                                        Body=json.dumps(model_spec),
                                                        Bucket="w210-poverty-mapper",
                                                        Key="modeling/model_specs/" + filename + ".json"
                                                )

                                        
def get_model_run_file(s3_filename, split_name_list, num_classes_list, bin_method_list, pretrained_list, freeze_layers_list, epoch_list, learning_rate_list, gamma_list, step_size_list, batch_size_list, num_workers_list):
    """
    Take an s3 filename and lists of model spec options, check if
    a model spec exists for each combination of model spec options, 
    and save a .json file with a list of model_spec.jsons to s3.
    
    s3_filename:          string filename to use for the model run file
    split_name_list:      list of string names of data splits from data_splits.json
    num_classes:          list of integer number of classes
    bin_method_list:      list of binning methods ("within", "across", "within_even", "across_even")
    pretrained_list:      list of string names of pytorch pretrained models
    freeze_layers_list:   list of options for whether or not to freeze layers ("yes, "no")
    epoch_list:           list of integer epoch values
    learning_rate_list:   list of float learning rate values  
    gamma_list:           list of integer gamma values
    step_size_list:       list of integer step sizes 
    batch_size_list:      list of integer batch sizes
    num_workers_list:     list of integer number of workers    
    """
    
    # Set boto3  client
    s3 = boto3.client("s3")
    
    # Create model run file dictionary
    model_run_file = {"model_specs": []}
    
    for split_name in split_name_list:
        for num_classes in num_classes_list:
            for bin_method in bin_method_list:
                for pretrained in pretrained_list:
                    for option in freeze_layers_list:
                        for epochs in epoch_list:
                            for learning_rate in learning_rate_list:
                                for gamma in gamma_list:
                                    for step_size in step_size_list:
                                        for batch_size in batch_size_list:
                                            for num_workers in num_workers_list:

                                                # Get filename
                                                spec_items = [split_name, str(num_classes), bin_method,
                                                              pretrained, option, str(epochs), str(learning_rate), str(gamma), str(step_size), str(batch_size), str(num_workers)]
                                                filename = "_".join(spec_items)

                                                # Check if model spec exists
                                                try:
                                                    s3.head_object(Bucket="w210-poverty-mapper", Key="modeling/model_specs/" + filename + ".json")
                                                except botocore.exceptions.ClientError as error:
                                                    print("{} model spec does not exist".format(filename + ".json"))
                                                    raise error

                                                # Check if results exist
                                                result = s3.list_objects(Bucket="w210-poverty-mapper", Prefix="modeling/results/" + filename + "_results.json")
                                                #print(result)
                                                if 'Contents' in result:
                                                    raise Exception("{} results already exist".format(filename + ".json"))

                                                # Append model spec json filename
                                                model_run_file["model_specs"].append("s3://w210-poverty-mapper/modeling/model_specs/" + filename + ".json")

    # Write model run file to s3
    s3.put_object(
         Body=json.dumps(model_run_file),
         Bucket="w210-poverty-mapper",
         Key="modeling/model_run_files/" + s3_filename + ".json"
    )