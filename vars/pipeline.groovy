// Pipeline stage definitions for the ML pipeline

def runDataAlignment(String imageName, String workspace) {
    echo 'Running data alignment pipeline...'
    def docker = load 'vars/docker.groovy'
    docker.runCommand(
        imageName,
        'python Scripts/data_alignment.py && mkdir -p /workspace/Data_sets && cp /app/Data_sets/train_dataset.csv /workspace/Data_sets/',
        [
            "${workspace}": '/workspace',
            "${workspace}/models": '/app/models',
            "${workspace}/reports": '/app/reports'
        ],
        '/app'
    )
    echo 'Data alignment completed'
}

def runModelBenchmarking(String imageName, String workspace) {
    echo 'Running model benchmarking...'
    def docker = load 'vars/docker.groovy'
    docker.runCommand(
        imageName,
        '''python Scripts/benchmark_models.py && \
mkdir -p /workspace/models /workspace/reports && \
cp -r /app/models/. /workspace/models/ && \
cp -r /app/reports/. /workspace/reports/''',
        [
            "${workspace}": '/workspace',
            "${workspace}/Data_sets": '/app/Data_sets',
            "${workspace}/models": '/app/models',
            "${workspace}/reports": '/app/reports'
        ],
        '/app'
    )
    echo 'Model benchmarking completed'
}

def generateVisualizations(String imageName, String workspace) {
    echo 'Generating performance visualizations...'
    def docker = load 'vars/docker.groovy'
    // Export reports via /workspace mount (reliable with Jenkins-in-Docker).
    docker.runCommand(
        imageName,
        '''python Scripts/visualize_results.py && \
mkdir -p /workspace/reports && \
cp -r /app/reports/. /workspace/reports/''',
        [
            "${workspace}": '/workspace',
            "${workspace}/Data_sets": '/app/Data_sets',
            "${workspace}/models": '/app/models',
            "${workspace}/reports": '/app/reports'
        ],
        '/app'
    )
    echo 'Visualizations generated'
}

def archiveArtifacts(String workspace, String outputDir) {
    echo 'Archiving artifacts to VM...'
    sh """
        set -e
        mkdir -p ${outputDir}
        cp -r ${workspace}/models ${outputDir}/
        cp -r ${workspace}/reports ${outputDir}/
        if [ -f ${workspace}/reports/performance_chart.png ]; then
            cp ${workspace}/reports/performance_chart.png ${outputDir}/performance_chart.png
        else
            echo "WARNING: performance_chart.png missing under reports/"
            ls -la ${workspace}/reports/ || true
            exit 1
        fi
        cp ${workspace}/config/benchmark_config.yaml ${outputDir}/
        echo "Artifacts archived to ${outputDir}"
    """
    archiveArtifacts artifacts: 'models/**, reports/**, config/benchmark_config.yaml',
                 allowEmptyArchive: false
}

return this
