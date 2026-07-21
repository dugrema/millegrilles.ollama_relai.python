pipeline {
    agent { label 'x86_64' }

    parameters {
        string(defaultValue: 'master', name: 'BRANCH')
        string(defaultValue: '2026.3', name: 'VERSION')
        string(defaultValue: 'jenkins-maple', name: 'CREDENTIALS_ID')
        string(defaultValue: 'ssh://git.maple.maceroc.com/git/millegrilles.ollama_relai.python.git', name: 'GIT_URL')
        string(defaultValue: 'registry.millegrilles.com:5000/millegrilles/ollama_relai_python', name: 'DOCKER_IMAGE')
    }

    environment {
        VBUILD="${VERSION}.${BUILD_NUMBER}"
        DOCKER_IMAGE="${params.DOCKER_IMAGE}"
    }

    stages {
        stage('docker build x86_64') {
            steps {
                checkout scmGit(branches: [[name: params.BRANCH]], extensions: [submodule(recursiveSubmodules: true, reference: '')], userRemoteConfigs: [[credentialsId: params.CREDENTIALS_ID, url: params.GIT_URL]])

                sh '''
                # Creer image docker
                docker build -t ${DOCKER_IMAGE}:${VBUILD} --build-arg VBUILD=${VBUILD} .
                docker push ${DOCKER_IMAGE}:${VBUILD}
                '''
            }
        }
    }
}
