scalaVersion := "2.11.8"

scalacOptions ++= Seq("-unchecked","-deprecation","-feature")

libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"

libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.0"

assemblyOutputPath in assembly := file(s"bin/${name.value}.jar")

assemblyMergeStrategy in assembly := {
  case PathList(ps @ _*) if ps.last endsWith ".class" => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}
