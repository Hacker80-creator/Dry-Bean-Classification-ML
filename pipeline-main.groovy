def call() {
    pipeline {
        agent any

        environment {
            DOCKER_IMAGE = 'bean-classification:${BUILD_NUMBER}'
            ARTIFACTORY_URL = 'http://your-jfrog-artifactory-url/artifactory'
            ARTIFACTORY_REPO = 'ml-models'
            ARTIFACTORY_CREDENTIALS = 'jfrog-credentials'
            OUTPUT_DIR = '/tmp/bean-classification-output'
        }

        stages {
            stage('Checkout') {
                steps {
                    checkout scm
                    echo 'Checked out source code'
                }
            }

            stage('Build Docker Image') {
                steps {
                    script {
                        docker.buildImage(env.DOCKER_IMAGE)
                    }
                }
            }

            stage('Run Data Alignment') {
                steps {
                    script {
                        pipeline.runDataAlignment(env.DOCKER_IMAGE, env.WORKSPACE)
                    }
                }
            }

            stage('Run Model Benchmarking') {
                steps {
                    script {
                        pipeline.runModelBenchmarking(env.DOCKER_IMAGE, env.WORKSPACE)
                    }
                }
            }

            stage('Generate Visualizations') {
                steps {
                    script {
                        pipeline.generateVisualizations(env.DOCKER_IMAGE, env.WORKSPACE)
                    }
                }
            }

            stage('Archive Artifacts to VM') {
                steps {
                    script {
                        pipeline.archiveArtifacts(env.WORKSPACE, env.OUTPUT_DIR)
                    }
                }
            }

            stage('Upload to JFrog Artifactory') {
                steps {
                    script {
                        pipeline.uploadToArtifactory(env.ARTIFACTORY_URL, env.ARTIFACTORY_REPO, env.BUILD_NUMBER, env.WORKSPACE)
                    }
                }
            }

            stage('Cleanup') {
                steps {
                    script {
                        docker.removeImage(env.DOCKER_IMAGE)
                    }
                }
            }
        }

        post {
            success {
                echo 'Pipeline completed successfully!'
                echo 'Artifacts are available in VM at: ${OUTPUT_DIR}'
                echo 'Artifacts uploaded to JFrog Artifactory at: ${ARTIFACTORY_URL}/${ARTIFACTORY_REPO}/bean-classification/${BUILD_NUMBER}/'
            }
            failure {
                echo 'Pipeline failed. Check logs for details.'
            }
            always {
                cleanWs()
            }
        }
    }
}
