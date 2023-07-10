# Sheet pile project

Before submitting your Argo workflow, make sure you have the following prerequisites in place:

Argo CLI: Install the Argo command-line interface (CLI) on your local machine. You can find installation instructions in the Argo documentation.
Kubernetes Cluster: Set up a Kubernetes cluster where you plan to run your Argo workflow. Ensure that you have the necessary permissions and access to the cluster.
Kubernetes Cluster: Set up a Kubernetes cluster where you plan to run your Argo workflow. Ensure that you have the necessary permissions and access to the cluster.

## Workflow Preparation

Follow these steps to prepare your Argo workflow for submission:

Workflow YAML: Create a YAML file that describes your Argo workflow. Make sure it includes all the necessary specifications, such as the templates, parameters, and steps required to execute your workflow. Refer to the Argo documentation for detailed instructions on creating a workflow YAML file.
Docker Images: If your workflow relies on custom Docker images, ensure that these images are available and accessible from the Kubernetes cluster. Push your Docker images to a container registry, such as Docker Hub or a private registry, and make sure the necessary credentials are set up for pulling these images during workflow execution.

## Workflow Submission

To submit your Argo workflow, follow these steps:

Validate Workflow: Validate your workflow YAML using the Argo CLI. Run the following command in your terminal:

argo lint <path_to_workflow_yaml>

This command checks for syntax errors and potential issues in your workflow definition.

Submit Workflow: Use the Argo CLI to submit your workflow to the Kubernetes cluster. Run the following command:

argo submit --watch <path_to_workflow_yaml> -n argo

The --watch flag allows you to monitor the progress of your workflow in real-time.

Monitor Workflow: After submitting the workflow, you can monitor its progress using the Argo CLI or the Argo web interface. To view the workflow status, run the following command:

argo get <workflow_name> -n argo

This command displays information such as the workflow status, individual step statuses, and logs.
