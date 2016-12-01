package jp.ac.gakushuin.cc.tk.scwc

package object utils {

  def time[A](f: => A) = {
    System.gc
    val startTime = System.nanoTime
    val result    = f
    val endTime   = System.nanoTime
    (result, (endTime - startTime)*1e-6)
  }

  import scala.collection.mutable.{OpenHashMap, MapLike}

  // for nested map
  class NestedMap[K, V](defaultValue: => V) extends OpenHashMap[K, V] 
      with MapLike[K, V, NestedMap[K, V]] {
    override def empty = new NestedMap[K, V](defaultValue)
    override def default(key: K): V = {
      val result = this.defaultValue
      this(key) = result
      result
    }
  }
  object NestedMap {
    def apply[K] = new Factory[K]
    class Factory[K] {
      def apply[V](defaultValue: => V) = new NestedMap[K, V](defaultValue)
    }
  }

}
