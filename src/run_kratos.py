# This is a sample Python script.

import sys
import os
import boto3
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# point to local Kratos installation
# sys.path.append(r"D:\Kratos_general\Kratos_build_version")
sys.path.append(r"/app/Kratos_linux")

import KratosMultiphysics as Kratos
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo
import KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis as analysis_geo

def get_nodal_variable(simulation, variable):

    nodes = simulation._list_of_output_processes[0].model_part.Nodes
    values = [node.GetSolutionStepValue(variable) for node in nodes]

    return values


def get_displacement_top_node(analysis):
    displacements = get_nodal_variable(analysis, Kratos.DISPLACEMENT)
    return displacements[-1][1]


def run_one_file_and_post_result(file_path):
    os.chdir(file_path)
    id = int(file_path.split("_")[-1])
    parameter_file_name = os.path.join(file_path, 'ProjectParameters.json')
    with open(parameter_file_name, 'r') as parameter_file:
        parameters_stage = Kratos.Parameters(parameter_file.read())
    model = Kratos.Model()
    analysis = analysis_geo.GeoMechanicsAnalysis(model, parameters_stage)
    analysis.Run()
    displacement = get_displacement_top_node(analysis)
    print("Displacement: " + str(displacement))
    # update database

    # Connect to the database
    conn = psycopg2.connect(
        host=os.environ.get("HOST"),
        dbname=os.environ.get("DBNAME"),
        user=os.environ.get("USERNAME"),
        password=os.environ.get("PASSWORD"),
        port=5432
    )

    c = conn.cursor()
    # update output in database based on directory name
    
    c.execute(f"UPDATE inputs_outputs SET displacement_out={displacement} WHERE id={id}")
    conn.commit()
    conn.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_dotenv()
    # run all files in the run directory
    run_directory = "/input"
    # path_database = "D:/SITO_sheet_piles/inputs_outputs.db"
    # list folders in run directory
    folders_to_run = os.listdir(run_directory)
    print(folders_to_run)
    # loop over folders
    for folder in folders_to_run:
        # run test
        run_one_file_and_post_result(os.path.join(run_directory, folder))



