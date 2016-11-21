scalaVersion := "2.11.8"

scalacOptions ++= Seq("-unchecked","-deprecation","-feature")

libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"

libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.0"  % "provided"

