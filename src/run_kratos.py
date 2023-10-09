from shetpilegenerator.calculation import run_multistage_calculation
from shetpilegenerator.output_utils import post_process
import pickle

if __name__ == '__main__':
    # open sqlite database and loop over the values
    total_models = 15
    failed_run_indexes = []
    for id in range(1, total_models):
        try:
            directory = f'src/results_RF_2/{id}'
            run_multistage_calculation(directory, 2)
        except Exception as e:
            print(e)
            failed_run_indexes.append(id)
    print(f"Failed runs: {failed_run_indexes}")
    print(f"Failed number: {len(failed_run_indexes)}")
    # perform the post processing
    for id in range(1, total_models):
        if id in failed_run_indexes:
            continue
        directory = f"D:/sheetpile/src/results_RF_2/{id}" 
        # open pickle file
        gmsh_to_kratos = pickle.load(open(f"{directory}/model.p", "rb"))
        post_process(2, 2.0, gmsh_to_kratos, save=True, directory=directory, file_name="stage_2.png")

