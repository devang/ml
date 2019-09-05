/* Copyright 2019 Devang Mistry. All rights reserved.
 * You may not remove this notice, but you may add
 * your name if you've made software modifications.
 * Software provided as-is. Available under the
 * terms of the AGPL v3 license available here:
 * https://www.gnu.org/licenses/agpl-3.0.en.html
 */

package jigsaw

import org.scalatest.{FlatSpec, Matchers}

class LibrarySuite extends FlatSpec with Matchers {

  "Jigsaw NB Pipeline" should "run" in {
    JigsawNB.jigsaw()
  }

  def testCat(): Unit = {
    assert(Utils.getCat(1,0,0,0,0,0) == 1)
    assert(Utils.getCat(0,0,0,0,0,1) == 32)
    assert(Utils.getCat(1,1,0,1,0,1) == 43)
    assert(Utils.getCat(1,1,1,1,1,1) == 63)
    assert(Utils.getCat(0,0,0,0,0,0) == 0)
  }

  def testCl() : Unit = {
    assert(Utils.getClasses(4) sameElements Array(3.0))
    assert(Utils.getClasses(Utils.getCat(0,1,0,0,0,1)) sameElements Array(2.0, 6))
    assert(Utils.getClasses(Utils.getCat(1,1,1,1,1,1)) sameElements Array(1, 2.0, 3, 4, 5, 6))
    assert(Utils.getClasses(Utils.getCat(0,1,0,0,0,0)) sameElements Array(2.0))
    assert(Utils.getClasses(Utils.getCat(0,0,0,0,0,0)) sameElements Array[Double](0))
  }

}
