package org.apache.spark.ml.param


import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import scala.collection.mutable

class HashMapParam(parent: Params, name: String, doc: String, isValid: mutable.HashMap[String, String] => Boolean)
  extends Param[mutable.HashMap[String, String]](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, ParamValidators.alwaysTrue)

  /** Creates a param pair with a `java.util.List` of values (for Java and Python). */
  def w(value: java.util.HashMap[java.lang.String, java.lang.String]): ParamPair[mutable.HashMap[String, String]] =
      w(value)

  override def jsonEncode(value: mutable.HashMap[String, String]): String = {
    import org.json4s.JsonDSL._
    compact(render(value.toSeq))
  }

  override def jsonDecode(json: String): mutable.HashMap[String, String] = {
    implicit val formats = DefaultFormats
    parse(json).extract[mutable.HashMap[String, String]]
  }
}