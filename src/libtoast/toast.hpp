/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_HPP
#define TOAST_HPP

#include <mpi.h>


namespace toast {

    void init ( int argc, char * argv[] );

    void finalize ( );

}

#include <toast/math.hpp>

#endif

