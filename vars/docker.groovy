// Docker operations for the ML pipeline

def buildImage(String imageName, String contextDir = '.') {
    echo "Building Docker image: ${imageName}"
    sh "docker build -t ${imageName} ${contextDir}"
    echo "Docker image built successfully"
}

def runCommand(String imageName, String command, Map volumeMounts = [:]) {
    def volumeArgs = volumeMounts.collect { k, v -> "-v ${k}:${v}" }.join(' ')
    echo "Running command in container: ${command}"
    sh """
        docker run --rm ${volumeArgs} \
            ${imageName} \
            ${command}
    """
}

def removeImage(String imageName) {
    echo "Removing Docker image: ${imageName}"
    sh "docker rmi ${imageName} || true"
    echo "Docker image removed"
}

return this
