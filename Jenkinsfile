pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'bean-classification:${BUILD_NUMBER}'
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
                    def docker = load 'vars/docker.groovy'
                    docker.buildImage(env.DOCKER_IMAGE)
                }
            }
        }

        stage('Run Data Alignment') {
            steps {
                script {
                    def pipeline = load 'vars/pipeline.groovy'
                    pipeline.runDataAlignment(env.DOCKER_IMAGE, env.WORKSPACE)
                }
            }
        }

        stage('Run Model Benchmarking') {
            steps {
                script {
                    def pipeline = load 'vars/pipeline.groovy'
                    pipeline.runModelBenchmarking(env.DOCKER_IMAGE, env.WORKSPACE)
                }
            }
        }

        stage('Generate Visualizations') {
            steps {
                script {
                    def pipeline = load 'vars/pipeline.groovy'
                    pipeline.generateVisualizations(env.DOCKER_IMAGE, env.WORKSPACE)
                }
            }
        }

        stage('Archive Artifacts to VM') {
            steps {
                script {
                    def pipeline = load 'vars/pipeline.groovy'
                    pipeline.archiveArtifacts(env.WORKSPACE, env.OUTPUT_DIR)
                }
            }
        }

        stage('Cleanup') {
            steps {
                script {
                    def docker = load 'vars/docker.groovy'
                    docker.removeImage(env.DOCKER_IMAGE)
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
            echo 'Artifacts are available in VM at: ${OUTPUT_DIR}'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
        always {
            cleanWs()
        }
    }
}
