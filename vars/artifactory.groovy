// JFrog Artifactory operations for artifact management

def uploadFile(String artifactoryUrl, String repo, String buildNumber, String localPath, String targetPath, String credentialsId = 'jfrog-credentials') {
    echo "Uploading ${localPath} to Artifactory"
    withCredentials([usernamePassword(
        credentialsId: credentialsId,
        usernameVariable: 'ARTIFACTORY_USER',
        passwordVariable: 'ARTIFACTORY_PASSWORD'
    )]) {
        sh """
            curl -u ${ARTIFACTORY_USER}:${ARTIFACTORY_PASSWORD} \
                -T ${localPath} \
                "${artifactoryUrl}/${repo}/bean-classification/${buildNumber}/${targetPath}"
        """
    }
    echo "File uploaded successfully"
}

def uploadArtifacts(String artifactoryUrl, String repo, String buildNumber, Map artifacts, String credentialsId = 'jfrog-credentials') {
    echo "Uploading ${artifacts.size()} artifacts to Artifactory"
    artifacts.each { localPath, targetPath ->
        uploadFile(artifactoryUrl, repo, buildNumber, localPath, targetPath, credentialsId)
    }
    echo "All artifacts uploaded to ${artifactoryUrl}/${repo}/bean-classification/${buildNumber}/"
}

def getArtifactUrl(String artifactoryUrl, String repo, String buildNumber, String artifactName) {
    return "${artifactoryUrl}/${repo}/bean-classification/${buildNumber}/${artifactName}"
}

return this
