## using this all over
import glob
import tqdm
import json
import os
from sklearn.metrics import classification_report, accuracy_score


def clean(dirnames=[None]):
    """
    A method for cleaning a collection of files
    """
    for el in dirnames:
        os.system(f"rm -rf {el}")
        print(f"Removed: {el}")


def make_a_dir(dirname):
    """
    A method for directory making.
    """
    try:
        os.mkdir(dirname)
        print(f"Made {dirname} folder.")
    except:
        pass


def write_all_jobs(all_jobs, raw_job_dir="raw_jobs"):
    """
    A method which writes all generated jobs.
    """
    make_a_dir(raw_job_dir)

    fx = open(f"./{raw_job_dir}/raw_jobs.txt", "w+")
    fx.write("\n".join(all_jobs))
    fx.close()

    print("Wrote all raw jobs")


def split_jobs(folder="raw_jobs", bsize=4):
    """
    A method which splits the jobs into batches (executed separately)
    """
    generate_splitjobs = f"""
    cd {folder};
    split -l {bsize} raw_jobs.txt;
    rm raw_jobs.txt;
    cd ..;
    """

    os.system(generate_splitjobs)
    print(f"Job batches ({bsize}) created.")


def generate_xrsl_jobs(data,
                       raw_job_folder="raw_jobs",
                       cluster_xrsl_folder="cluster_xrsl"):
    """
    A method which generates the xrsl job specifications
    """
    make_a_dir(cluster_xrsl_folder)

    for enx, job_batch in tqdm.tqdm(enumerate(
            glob.glob(f"{raw_job_folder}/*"))):
        px = "../" + job_batch
        dnew = data.replace("PLACEHOLDER", px)
        dnew.replace("JIDX", str(enx))
        with open(
                f"{cluster_xrsl_folder}/" + job_batch.split("/")[1] + ".xrsl",
                "w") as text_file:
            text_file.write(dnew)


def output_classification_results(predictions,
                                  test_classes,
                                  path=None,
                                  print_out=False,
                                  model_spec=None):
    """
    A method which creates the report JSON from the final set of predictions.
    """
    report_classification = classification_report(test_classes,
                                                  predictions,
                                                  output_dict=True)

    try:
        test_classes = test_classes.tolist()
    except Exception:
        pass

    try:
        predictions = predictions.tolist()
    except Exception:
        pass

    report_classification['real'] = test_classes
    report_classification['predictions'] = predictions
    report_classification['model_specification'] = model_spec
    report_classification['accuracy'] = accuracy_score(test_classes,
                                                       predictions)

    try:
        with open(path, 'w') as f:
            json.dump(report_classification, f)

    except:
        pass
