plugins {
    scala
}

repositories {
    mavenCentral()
    jcenter()
    mavenLocal()
    google()
    maven {
        url = uri("http://dl.bintray.com/spark-packages/maven/")
    }
}

dependencies {
    implementation("org.scala-lang:scala-library:2.12.9")
    implementation("org.apache.spark:spark-mllib_2.12:2.4.4")
    implementation("org.apache.spark:spark-streaming_2.12:2.4.4")
    implementation("com.lihaoyi:ammonite-ops_2.12:1.6.9")
    implementation("org.testng:testng:6.14.3")
    implementation("org.scalatest:scalatest_2.12:3.0.8")
    implementation("com.intel.analytics.bigdl:bigdl:0.10.0-SNAPSHOT")
    implementation("com.github.fommil.netlib:netlib-native_system-linux-x86_64:1.+")
    implementation("org.mlflow:mlflow-client:1.2.0")
    implementation("org.plotly-scala:plotly-render_2.12:0.7.0")
    implementation("edu.stanford.nlp:stanford-corenlp:3.9.2")
    implementation("org.typelevel:cats-effect_2.12:1.4.0")
    implementation("org.typelevel:frameless-ml_2.12:0.8.0")
    implementation("org.typelevel:frameless-cats_2.12:0.8.0")
//    implementation("databricks:tensorframes:0.7.0-s_2.11")
}

tasks.jar {
    from(configurations.compile.map { configuration ->
        configuration.asFileTree.fold(files().asFileTree) { collection, file ->
            if (file.isDirectory) collection else collection.plus(zipTree(file))
        }
    })
}

tasks.test {
    useTestNG()
}
