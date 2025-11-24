// Define mesh, run with the command "gmsh symmetric_mesh.geo -2"

// Super coarse mesh 4 (806 nodes 1682 elements)
DefineConstant[ lc  = {0.6}];
DefineConstant[ lc1  = {0.14}];

// Coarse mesh 3 (1645 nodes 3387 elements)
//DefineConstant[ lc  = {0.4}];
//DefineConstant[ lc1  = {0.10}];

// Medium mesh 2 (3808 nodes 7760 elements)
//DefineConstant[ lc  = {0.25}];
//DefineConstant[ lc1  = {0.07}];

// Fine mesh 1 (7217 nodes 14631 elements)
//DefineConstant[ lc  = {0.18}];
//DefineConstant[ lc1  = {0.05}];

// Fine mesh 0 (16843 nodes 33989 elements)
//DefineConstant[ lc  = {0.12}];
//DefineConstant[ lc1  = {0.03}];

// Circle points
DefineConstant[ radius  = {0.5}];
Point(1) = {0, 0, 0, lc1};
Point(2) = {-radius, 0, 0, lc1};
Point(3) = {0, radius, 0, lc1};


// Circle lines
Circle(10) = {3, 1, 2};

// Rectangle points
Point(4) = {-5, 2.5, 0, lc};
Point(5) = {-5, 0, 0, lc};
Point(6) = {15, 0, 0, lc};
Point(7) = {15, 2.5, 0, lc};

// Rectangle lines
Line(7) = {6, 7};
Line(8) = {7, 4};
Line(9) = {4, 5};

// Perform symmetry
Symmetry {0, 1, 0, 0} { Duplicata { Curve{9}; Curve{8}; Curve{7}; } }
Symmetry {1, 0, 0, 0.0} { Duplicata { Curve{10}; } }
Symmetry {0, 1, 0, 0.0} { Duplicata { Curve{14}; } }
Symmetry {-1, 0, 0, 0.0} { Duplicata { Curve{15}; } }
Line(17) = {5, 2};
Line(18) = {15, 6};
Curve Loop(1) = {8, 9, 17, -10, 14, 18, 7};
Plane Surface(1) = {1};
Symmetry {0, 1, 0, 0} {
  Duplicata { Surface{1}; }
}

// Remove central lines
Recursive Delete {
  Curve{17}; Curve{18}; 
}

// Create Physical curves
Physical Curve(10) = {11, 9};
Physical Curve(12) = {8, 12};
Physical Curve(11) = {7, 13};
Physical Curve(13) = {10, 16, 14, 15};
Physical Surface(1) = {1, 19};

// Refinement rectangle
//lc2 = 0.001
//Point(41) = {5, -1, 0, lc2};
//Point(42) = {15, -1, 0, lc2};
//Point(43) = {15, 1, 0, lc2};
//Point(44) = {5, 1, 0, lc2};
//Line(41) = {41, 42};
//Line(42) = {42, 43};
//Line(43) = {43, 44};
//Line(44) = {44, 41};
//Line Loop(45) = {43, 44, 41, 42};
//Plane Surface(46) = {45};
//Point{41} In Surface{1};
//Point{42} In Surface{1};
//Plane Surface(60) = {};
//Point{43} In Loop{1};
//Point{44} In Loop{1};



