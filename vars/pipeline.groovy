// Pipeline stage definitions for the ML pipeline

def runDataAlignment(String imageName, String workspace) {
    echo 'Running data alignment pipeline...'
    sh "ls -la ${workspace}"
    sh "ls -la ${workspace}/Scripts || echo 'Scripts directory not found'"
    def docker = load 'vars/docker.groovy'
    docker.runCommand(imageName, 'python Scripts/data_alignment.py', [
        "${workspace}": '/workspace'
    ])
    echo 'Data alignment completed'
}

def runModelBenchmarking(String imageName, String workspace) {
    echo 'Running model benchmarking...'
    def docker = load 'vars/docker.groovy'
    docker.runCommand(imageName, 'python Scripts/benchmark_models.py', [
        "${workspace}": '/workspace'
    ])
    echo 'Model benchmarking completed'
}

def generateVisualizations(String imageName, String workspace) {
    echo 'Generating performance visualizations...'
    def docker = load 'vars/docker.groovy'
    docker.runCommand(imageName, 'python Scripts/visualize_results.py', [
        "${workspace}": '/workspace'
    ])
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

def uploadToArtifactory(String artifactoryUrl, String repo, String buildNumber, String workspace) {
    echo 'Uploading artifacts to JFrog Artifactory...'
    def artifactory = load 'vars/artifactory.groovy'
    
    def artifacts = [
        "${workspace}/models/best_model.joblib": "best_model.joblib",
        "${workspace}/models/model_metadata.json": "model_metadata.json",
        "${workspace}/reports/benchmark_results.csv": "benchmark_results.csv",
        "${workspace}/reports/best_model_metrics.json": "best_model_metrics.json",
        "${workspace}/performance_chart.png": "performance_chart.png",
        "${workspace}/config/benchmark_config.yaml": "benchmark_config.yaml"
    ]
    
    artifactory.uploadArtifacts(artifactoryUrl, repo, buildNumber, artifacts)
    echo "All artifacts uploaded to JFrog Artifactory"
}

return this
