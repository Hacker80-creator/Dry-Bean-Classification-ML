// Pipeline stage definitions for the ML pipeline

def runDataAlignment(String imageName, String workspace) {
    echo 'Running data alignment pipeline...'
    def docker = load 'vars/docker.groovy'
    // Run alignment with container's built-in Data_sets, then export generated CSV
    // to the Jenkins workspace in the same container lifecycle.
    docker.runCommand(imageName, 'python Scripts/data_alignment.py && mkdir -p /workspace/Data_sets && cp /app/Data_sets/train_dataset.csv /workspace/Data_sets/', [
        "${workspace}": '/workspace',
        "${workspace}/models": '/app/models',
        "${workspace}/reports": '/app/reports'
    ], '/app')
    echo 'Data alignment completed'
}

def runModelBenchmarking(String imageName, String workspace) {
    echo 'Running model benchmarking...'
    def docker = load 'vars/docker.groovy'
    docker.runCommand(imageName, 'python Scripts/benchmark_models.py', [
        "${workspace}/Data_sets": '/app/Data_sets',
        "${workspace}/models": '/app/models',
        "${workspace}/reports": '/app/reports'
    ], '/app')
    echo 'Model benchmarking completed'
}

def generateVisualizations(String imageName, String workspace) {
    echo 'Generating performance visualizations...'
    def docker = load 'vars/docker.groovy'
    // Run visualization and copy chart to workspace in same command
    docker.runCommand(imageName, 'python Scripts/visualize_results.py && cp /app/performance_chart.png /workspace/', [
        "${workspace}": '/workspace',
        "${workspace}/Data_sets": '/app/Data_sets',
        "${workspace}/reports": '/app/reports'
    ], '/app')
    echo 'Visualizations generated'
}

def archiveArtifacts(String workspace, String outputDir) {
    echo 'Archiving artifacts to VM...'
    sh """
        mkdir -p ${outputDir}
        cp -r ${workspace}/models ${outputDir}/
        cp -r ${workspace}/reports ${outputDir}/
        cp ${workspace}/performance_chart.png ${outputDir}/
        cp ${workspace}/config/benchmark_config.yaml ${outputDir}/
        echo "Artifacts archived to ${outputDir}"
    """
    archiveArtifacts artifacts: 'models/**, reports/**, performance_chart.png, config/benchmark_config.yaml', 
                 allowEmptyArchive: false
}

return this
