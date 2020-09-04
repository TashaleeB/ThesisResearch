%-------------------------------------------------------------------
%LIST OF PARAMETERS: All units are in Microns
%-------------------------------------------------------------------
pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286

%Tile
TileWidth = 5000;
TileHeight = 5000*sqrt(3)/2;
TileCenterPosition = -1000*i;
TileFeedlineWidth = 250;

%Main Feedline Properties
mFeedlineLength = 3502;
mFeedlineWidth = 250;

 
%Absorber - X and Absorber - Y
ABSWidth = 3;
ABSLengthX1 = 300;
ABSLengthX2= 300;
ABSLengthY1 = 300;
ABSLengthY2= 300;
ABSSpacing = 2;
ABSPolarizationX = 'X';
ABSPolarizationY = 'Y';
ABSRotationX = 0;
ABSTopOuterYRotation = 90;
ABSBottomOuterYRotation = 270;
ABSTopInnerYRotation = 90;
ABSBottomInnerYRotation = 270;
ABSTopOuterXRotation = 90;
ABSBottomOuterXRotation = 270;
ABSTopInnerXRotation = 90;
ABSBottomInnerXRotation = 270;
XABSArchStartAngle = 90;
XABSArchEndAngle = 270;
YABSTopArchStartAngle = 180;
YABSTopArchEndAngle = 360;
YABSBottomArchStartAngle = 0;
YABSBottomArchEndAngle = 180;
Angle2 = 180;
Angle1 = 0;
ABSArcWidth = 10;
ABSArcSpacing = 10;
ABSArcMedianOuterRadius = 275;
ABSArcMedianInnerRadius = 265;
ABSTopOuterFingerLength = ABSArcMedianOuterRadius;
ABSBottomOuterFingerLength = ABSArcMedianOuterRadius;
ABSTopInnerFingerLength = ABSArcMedianInnerRadius;
ABSBottomInnerFingerLength = ABSArcMedianInnerRadius;

 
%Connection line from Absorber to Feedline around IDC
ABStoIDCFeedlineTopLength = 132;
ABStoIDCFeedlineBottomLength = ABStoIDCFeedlineTopLength;
ABStoIDCFeedlineWidth = 20;
ABStoIDCFeedlineSpacing = 15;

 
%IDC
IDCLength = 1210;
IDCFingerWidth = 3;
IDCSpacing = IDCFingerWidth*2;
IDCFingerOverlap = 1200;
IDCEndGap = 10;
IDCTerminalWidth = 40;
IDCNumPairs = 80;
%Go back through and make the Code and---------
IDCTopShift = 10;% |
% --------------------------------------------
%changes To get rid of that constants
IDCSideShift = 2*IDCNumPairs*IDCFingerWidth + (IDCNumPairs*2-1)*IDCSpacing/2;
NumberofIDCFingersRemoved = 0;

 
%Feedline Around IDC
IDCTopFeedlineWidth = 50;
IDCBottomFeedlineWidth = IDCTopFeedlineWidth;
IDCSide2AandBSpacing = ABStoIDCFeedlineSpacing;
IDCSide1FeedlineLength = IDCLength+IDCTopShift;
IDCSide2AFeedlineLength = IDCSide1FeedlineLength/2 - IDCSide2AandBSpacing/2;
IDCSide2BFeedlineLength = IDCSide1FeedlineLength/2 - IDCSide2AandBSpacing/2;
IDCSide1FeedlineWidth = IDCTopFeedlineWidth;
IDCSide2AFeedlineWidth = IDCSide1FeedlineWidth;
IDCSide2BFeedlineWidth = IDCSide1FeedlineWidth;
SpaceBetweenIDCandIDCFeedline1 = 51;%Did not use this
SpaceBetweenIDCandIDCFeedline2 = 7;%Did not use this
IDCSafeSpace = (1-NumberofIDCFingersRemoved)*(IDCFingerWidth + 2*IDCSpacing);
IDCTopFeedlineLength = 8+ 10*(IDCFingerWidth+IDCSpacing)+IDCSideShift+IDCSide2AFeedlineWidth;
IDCBottomFeedlineLength = IDCTopFeedlineLength;
 
%Connection Line Between IDC and CPLIDC Line (Coupled IDC)
IDCtoCLPIDCFeedlineLength = 173;
IDCtoCLPIDCFeedlineWidth = 50;

 
%Coupler IDC
CPLIDCLength = 150;
CPLIDCFingerWidth = 5;
CPLIDCSpacing = 5;
CPLIDCFingerOverlap = 100;
CPLIDCEndGap = 2;
CPLIDCTerminalWidth = 20;
CPLIDCNumPairs = 15;
CPLIDCTopShift = 10;
CPLIDCSideShift = 2*CPLIDCNumPairs*CPLIDCFingerWidth + (CPLIDCNumPairs*2-1)*CPLIDCSpacing;
NumberofCPLIDCFingersRemoved = 0;

 
%Feedline around Coupled IDC
CPLIDCSafeSpace = (1-NumberofCPLIDCFingersRemoved)*(CPLIDCFingerWidth + 2*CPLIDCSpacing);
CPLIDCTopFeedlineLength = 4*CPLIDCFingerWidth+4*CPLIDCSpacing+ CPLIDCSideShift;
CPLIDCBottomFeedlineLength = CPLIDCTopFeedlineLength;
CPLIDCTopFeedlineWidth = 25;
CPLIDCBottomFeedlineWidth = 25;

 
%Conncection from Coupled IDC Bottom line to main Feedline (mFeedline)
CPLIDCBottomFeedlinetomFeedlineLength = IDCtoCLPIDCFeedlineLength;
CPLIDCBottomFeedlinetomFeedlineWidth = IDCtoCLPIDCFeedlineWidth;


%Spacing between each Resonator within one pixel
DistanceBetweenConnCPLIDCtomFeedline = 840;
DistanceBetweenCPLIDCtoCPLIDC= 840+46;
DistanceBetweenConnIDCtoCPLIDC = DistanceBetweenConnCPLIDCtomFeedline;
DistanceBetweenIDCtoIDC = 960+5; %+ 2*IDCTopFeedlineLength;

 
%-------------------------------------------------------------------

%                           Start Building Pixel

%-------------------------------------------------------------------
pixel.W = [];
W = [];

%-------------------------------------------------------------------
%Create Tile outline as a wire, in CW direction
%-------------------------------------------------------------------
W.layer = 'A'; %layer name of the wire
W.z = TileCenterPosition + [TileWidth/2 + TileHeight/2*i; +TileWidth/2 - TileHeight/2*i;-TileWidth/2 - TileHeight/2*i;-TileWidth/2 + TileHeight/2*i;+TileWidth/2 + TileHeight/2*i];
W.w = 1;%width of the wire
pixel.W = [pixel.W,W]; %add the wire to list

%-------------------------------------------------------------------
%Create Main Feedline as a wire
%-------------------------------------------------------------------
W.layer = 'A'; %layer name of the wire
W.z = [-TileWidth/2 - TileHeight/2*i;TileWidth/2 - TileHeight/2*i];%vertices of the wire
W.w = mFeedlineWidth;%width of the wire
pixel.W = [pixel.W,W]; %add the wire to list 

%-------------------------------------------------------------------
%Create Absorber - Y
%-------------------------------------------------------------------
W.layer = 'A';%Outer Top half semicircle
W.z = ABSArcMedianOuterRadius*exp([ABSTopOuterYRotation:1:Angle2]'*pi/180*i)+ (-15+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -(-2+ ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius) +ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W]

W.layer = 'A'; %Outer Bottom half semicircle
W.z = ABSArcMedianOuterRadius*exp([180:1:ABSBottomOuterYRotation]'*pi/180*i)+ (-15+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -(-2+ ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius) +ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Inner Top half semicircle
W.z = ABSArcMedianInnerRadius*exp([ABSTopInnerYRotation:1:Angle2]'*pi/180*i)+ (-15+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -( ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius) +ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W]

W.layer = 'A'; %Inner Bottom half semicircle
W.z = ABSArcMedianInnerRadius*exp([Angle2:1:ABSBottomInnerYRotation]'*pi/180*i)+ (-15+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -( ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius) +ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Top outer bar
W.z = [(-15+ ABSWidth/2+ABSArcMedianOuterRadius+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -11+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-31+ -ABSTopOuterFingerLength+ -ABSWidth+ABSArcMedianOuterRadius+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -11+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Bottom outer bar
W.z = [(-15+ -ABSWidth/2+ -ABSArcMedianOuterRadius+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -11+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(1+ ABSBottomOuterFingerLength+ABSWidth+ -ABSArcMedianOuterRadius+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -11+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Bar
W.z = [( -1.5*ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -11.5+ ABStoIDCFeedlineWidth/2+ ABStoIDCFeedlineTopLength+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);( ABStoIDCFeedlineSpacing+ -1.5*ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -11.5+ ABStoIDCFeedlineWidth/2+ ABStoIDCFeedlineTopLength+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Top inner bar
W.z = [(-15+ -ABSWidth/2+ABSArcMedianInnerRadius+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -6+ ABSWidth+ABSArcMedianInnerRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-31+ -ABSTopInnerFingerLength+ -ABSWidth/2+ABSArcMedianInnerRadius+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -6+ ABSWidth+ABSArcMedianInnerRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Bottom inner bar
W.z = [(-15 +ABSWidth/2+ -ABSArcMedianInnerRadius+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -6+ ABSWidth+ABSArcMedianInnerRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(1+ ABSBottomInnerFingerLength+ABSWidth+ -ABSArcMedianInnerRadius+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -6+ ABSWidth+ABSArcMedianInnerRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Top arch for Absorber Finger
W.z = (1+ABSWidth/2)*exp([YABSTopArchStartAngle:1:YABSTopArchEndAngle]'*pi/180*i)+ (-31+ -ABSTopInnerFingerLength+ -ABSWidth/2+ABSArcMedianInnerRadius+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ (ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)/2+ -9+ 0.69*ABSWidth+ABSArcMedianInnerRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Bottom arch for Absorber Finger
W.z = (1+ABSWidth/2)*exp([YABSBottomArchStartAngle:1:YABSBottomArchEndAngle]'*pi/180*i)+ (1+ ABSBottomInnerFingerLength+ABSWidth+ -ABSArcMedianInnerRadius+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ (ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)/2+ -9+ 0.69*ABSWidth+ABSArcMedianInnerRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create Absorber - X
%-------------------------------------------------------------------
W.layer = 'A';%Outer Top half semicircle
W.z = ABSArcMedianOuterRadius*exp([0:1:ABSTopOuterXRotation]'*pi/180*i)+ (-15+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ (-2+ ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)+ -(ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W]

W.layer = 'A'; %Outer Bottom half semicircle
W.z = ABSArcMedianOuterRadius*exp([ABSBottomOuterXRotation:1:360]'*pi/180*i)+ (-15+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ (-2+ ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)+ -(ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Inner Top half semicircle now changed to full half circle
W.z = ABSArcMedianInnerRadius*exp([0:1:ABSTopInnerXRotation]'*pi/180*i)+ (-15+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ (ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)+ -(ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W]

W.layer = 'A'; %Inner Bottom half semicircle
W.z = ABSArcMedianInnerRadius*exp([ABSBottomInnerXRotation:1:360]'*pi/180*i)+ (-15+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ (ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)+ -(ABSArcMedianOuterRadius+ ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Top outer bar
W.z = [(-15+ -ABSWidth/2+ABSArcMedianOuterRadius+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -(-15+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ -(-0.5*ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)+ -ABSWidth/2+ ABSArcMedianOuterRadius+ ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -(-15+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Bottom outer bar
W.z = [(-15 +ABSWidth/2+ -ABSArcMedianOuterRadius+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -(-15+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);((-15+ -ABSWidth+ ABSArcMedianOuterRadius-ABSArcMedianInnerRadius)+ ABSWidth+ -ABSArcMedianOuterRadius+ -ABStoIDCFeedlineWidth+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -(-15+ ABSWidth+ABSArcMedianOuterRadius+ 0*ABSWidth/2+ ABStoIDCFeedlineTopLength+IDCTopFeedlineLength)+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Bar
W.z = [( -1.5*ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ 11.5+ -ABStoIDCFeedlineWidth/2+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);( ABStoIDCFeedlineSpacing+ -1.5*ABStoIDCFeedlineWidth+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ 11.5+ -ABStoIDCFeedlineWidth/2+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; % X absorber Top finger
W.z = [(-21.5+ ABSWidth/2+ 0*ABStoIDCFeedlineWidth/2+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-21.5+ ABSWidth/2+ 0*ABStoIDCFeedlineWidth/2+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ (17.5+ -2*ABSArcMedianOuterRadius+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace))];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; % X absorber Bottom finger
W.z = [(-8.5+ -ABSWidth/2+ -0*ABStoIDCFeedlineWidth/2+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-8.5+ -ABSWidth/2+ -0*ABStoIDCFeedlineWidth/2+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ (17.5+ -2*ABSArcMedianOuterRadius+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace))];
W.w = ABSWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Arch for Absorber Finger
W.z = (1+ABSWidth/2)*exp([XABSArchStartAngle:1:XABSArchEndAngle]'*pi/180*i)+ (-15+ -ABStoIDCFeedlineSpacing/2+ 0*ABSWidth/2+ 0*ABStoIDCFeedlineWidth/2+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ (17.5+ -2*ABSArcMedianOuterRadius+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace));
W.w = ABSWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create connection line from Absorber to Feedline around IDC ~LEFT SIDE
%-------------------------------------------------------------------
W.layer = 'A'; %Top
W.z = [(-15+ ABStoIDCFeedlineWidth/2+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -21.5+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ ABStoIDCFeedlineWidth/2+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ ABStoIDCFeedlineTopLength+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABStoIDCFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; %Bottom
W.z = [(-15+ -ABStoIDCFeedlineWidth/2+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -21.5+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ -ABStoIDCFeedlineWidth/2+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ ABStoIDCFeedlineTopLength+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABStoIDCFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create connection line from Absorber to Feedline around IDC ~RIGHT SIDE
%-------------------------------------------------------------------
W.layer = 'A'; %Top
W.z = [(-15+ ABStoIDCFeedlineWidth/2+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ 21.5+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ ABStoIDCFeedlineWidth/2+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABStoIDCFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; %Bottom
W.z = [(-15+ -ABStoIDCFeedlineWidth/2+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ 21.5+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ -ABStoIDCFeedlineWidth/2+IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -ABStoIDCFeedlineTopLength+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = ABStoIDCFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create feedline around IDC as a wire  ~LEFT SIDE
%-------------------------------------------------------------------
W.layer = 'A'; %Bottom Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2)*i+ -(20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2)*i+IDCBottomFeedlineLength+ -(20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace)];
W.w = IDCBottomFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; %Top Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -(20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+IDCTopFeedlineLength+ -(20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = IDCTopFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; %Side 1 Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2+IDCBottomFeedlineWidth/2)*i+ -(7.5+21.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace/2-IDCBottomFeedlineWidth/2);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2+ IDCBottomFeedlineWidth/2+IDCSide1FeedlineLength)*i + -(7.5+21.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace/2-IDCBottomFeedlineWidth/2)];
W.w = IDCSide1FeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Side 2A Feedline
W.z = [(-15+ -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -21.5+ -IDCSide2BFeedlineWidth/2+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -21.5+ -IDCSide2AFeedlineWidth/2+ IDCTopFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = IDCSide2AFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Side 2B Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -21.5+ -IDCSide2BFeedlineWidth/2+ IDCBottomFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace);(-15+ IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -21.5+ -IDCSide2BFeedlineWidth/2+ IDCBottomFeedlineLength+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace)];
W.w = IDCSide2BFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create feedline around IDC as a wire  ~RIGHT SIDE
%-------------------------------------------------------------------
W.layer = 'A'; %Bottom Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2)*i+ (20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2)*i+ -IDCBottomFeedlineLength+ (20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace)];
W.w = IDCBottomFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; %Top Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ (20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ -IDCTopFeedlineLength+ (20.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = IDCTopFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A'; %Side 1 Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2+IDCBottomFeedlineWidth/2)*i+ (7.5+21.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace/2-IDCBottomFeedlineWidth/2);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth/2+ IDCBottomFeedlineWidth/2+IDCSide1FeedlineLength)*i + (7.5+21.5+ DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace/2-IDCBottomFeedlineWidth/2)];
W.w = IDCSide1FeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Side 2A Feedline
W.z = [(-15+ -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ 21.5+ IDCSide2BFeedlineWidth/2+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace);(-15+ -IDCSide2AFeedlineLength -IDCTopFeedlineWidth/2 -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCTopFeedlineWidth+IDCTopShift+IDCLength+IDCTopFeedlineWidth/2)*i+ 21.5+ IDCSide2AFeedlineWidth/2+ -IDCTopFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCTopFeedlineWidth+IDCSafeSpace)];
W.w = IDCSide2AFeedlineWidth;
pixel.W = [pixel.W,W];

W.layer = 'A';%Side 2B Feedline
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ 21.5+ IDCSide2BFeedlineWidth/2+ -IDCBottomFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace);(-15+ IDCSide2BFeedlineLength+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ 21.5+ IDCSide2BFeedlineWidth/2+ -IDCBottomFeedlineLength+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCBottomFeedlineWidth+IDCSafeSpace)];
W.w = IDCSide2BFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create connection Line Between IDC and CPLIDC Line (Coupled IDC) ~LEFT SIDE
%-------------------------------------------------------------------
W.layer = 'A';
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2)*i+ -(DistanceBetweenConnIDCtoCPLIDC/2+IDCtoCLPIDCFeedlineWidth/2);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength)*i+ -(DistanceBetweenConnIDCtoCPLIDC/2+IDCtoCLPIDCFeedlineWidth/2)];
W.w = IDCtoCLPIDCFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create connection Line Between IDC and CPLIDC Line (Coupled IDC) ~RIGHT SIDE
%-------------------------------------------------------------------
W.layer = 'A';
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2)*i+ (DistanceBetweenConnIDCtoCPLIDC/2+IDCtoCLPIDCFeedlineWidth/2);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength)*i+ (DistanceBetweenConnIDCtoCPLIDC/2+IDCtoCLPIDCFeedlineWidth/2)];
W.w = IDCtoCLPIDCFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create Top feedline around Coupled IDC ~LEFT SIDE
%-------------------------------------------------------------------
W.layer = 'A';
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength)*i + -(DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength)*i+ CPLIDCTopFeedlineLength+ -(DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace)];
W.w = CPLIDCTopFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create Bottom feedline around Coupled IDC ~LEFT SIDE
%-------------------------------------------------------------------
W.layer = 'A';
W.z = [(-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2)*i + -(DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace);(-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2)*i+ CPLIDCBottomFeedlineLength+ -(DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace)];
W.w = CPLIDCBottomFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create Top feedline around Coupled IDC ~RIGHT SIDE
%-------------------------------------------------------------------
W.layer = 'A';
W.z = [(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength)*i + (DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace);(-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength)*i+ -CPLIDCTopFeedlineLength+ (DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace)];
W.w = CPLIDCTopFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create Bottom feedline around Coupled IDC ~RIGHT SIDE
%-------------------------------------------------------------------
W.layer = 'A';
W.z = [(-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2)*i + (DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace);(-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2)*i+ -CPLIDCBottomFeedlineLength+ (DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift+CPLIDCSafeSpace)];
W.w = CPLIDCBottomFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create conncection from Coupled IDC Bottom line to main Feedline (mFeedline) ~LEFT SIDE
%-------------------------------------------------------------------
W=[];
W.layer = 'A';
W.z = [(-TileHeight/2+TileFeedlineWidth/2)*i - (DistanceBetweenConnCPLIDCtomFeedline/2+CPLIDCBottomFeedlinetomFeedlineWidth/2);-(DistanceBetweenConnCPLIDCtomFeedline/2+CPLIDCBottomFeedlinetomFeedlineWidth/2) + (-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength)*i];
W.w = CPLIDCBottomFeedlinetomFeedlineWidth;
pixel.W = [pixel.W,W];
 
%-------------------------------------------------------------------
%Create Top IDC fingers. There are a total of . ~LEFT SIDE
%-------------------------------------------------------------------
for u=1:IDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth +IDCTopShift)*i+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCSpacing/2)) + [0;IDCLength*i] + u*4*IDCFingerWidth;
    W(u).w = IDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list

%-------------------------------------------------------------------
%Create Bottom IDC fingers. There are a total of . ~LEFT SIDE
%-------------------------------------------------------------------
for u=1:IDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ -(DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCFingerWidth+IDCSpacing)) + [0;IDCLength*i] + u*4*IDCFingerWidth;
    W(u).w = IDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list
 
%-------------------------------------------------------------------
%Create Top coupler IDC fingers. There are a total of . ~LEFT SIDE
%-------------------------------------------------------------------
for u=1:CPLIDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2+CPLIDCBottomFeedlineWidth/2 +CPLIDCTopShift)*i + -(DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift-2*CPLIDCSpacing)) + [0;CPLIDCLength*i] + u*4*CPLIDCFingerWidth
    W(u).w = CPLIDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list

 
%-------------------------------------------------------------------
%Create Bottom coupler IDC fingers. There are a total of . ~LEFT SIDE
%-------------------------------------------------------------------
for u=1:CPLIDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2+CPLIDCBottomFeedlineWidth/2)*i+ -(DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift)) + [0;CPLIDCLength*i] + u*4*CPLIDCFingerWidth
    W(u).w = CPLIDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list

%-------------------------------------------------------------------
%Create conncection from Coupled IDC Bottom line to main Feedline (mFeedline) ~RIGHT SIDE
%-------------------------------------------------------------------
W=[];
W.layer = 'A';
W.z = [(-TileHeight/2+TileFeedlineWidth/2)*i+ (DistanceBetweenConnCPLIDCtomFeedline/2+CPLIDCBottomFeedlinetomFeedlineWidth/2);(-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength)*i+ (DistanceBetweenConnCPLIDCtomFeedline/2+CPLIDCBottomFeedlinetomFeedlineWidth/2)];
W.w = CPLIDCBottomFeedlinetomFeedlineWidth;
pixel.W = [pixel.W,W];

%-------------------------------------------------------------------
%Create Top coupler IDC fingers. There are a total of . ~RIGHT SIDE
%-------------------------------------------------------------------
for u=1:CPLIDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2+CPLIDCBottomFeedlineWidth/2 +CPLIDCTopShift)*i + (DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift-2*CPLIDCSpacing)) + [0;CPLIDCLength*i] + -u*4*CPLIDCFingerWidth
    W(u).w = CPLIDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list

%-------------------------------------------------------------------
%Create Bottom coupler IDC fingers. There are a total of . ~RIGHT SIDE
%-------------------------------------------------------------------
for u=1:CPLIDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCBottomFeedlineWidth/2+CPLIDCBottomFeedlineWidth/2)*i+ (DistanceBetweenCPLIDCtoCPLIDC/2+CPLIDCFingerWidth/2+CPLIDCSideShift)) + [0;CPLIDCLength*i] + -u*4*CPLIDCFingerWidth
    W(u).w = CPLIDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list

%-------------------------------------------------------------------
%Create Top IDC fingers. There are a total of . ~RIGHT SIDE
%-------------------------------------------------------------------
for u=1:IDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth +IDCTopShift)*i+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCSpacing/2)) + [0;IDCLength*i] + -u*4*IDCFingerWidth;
    W(u).w = IDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list

%-------------------------------------------------------------------
%Create Bottom IDC fingers. There are a total of . ~RIGHT SIDE
%-------------------------------------------------------------------
for u=1:IDCNumPairs
    W(u).layer = 'A'; %layer name of the wire
    W(u).z = ((-15+ -TileHeight/2+TileFeedlineWidth/2+CPLIDCBottomFeedlinetomFeedlineLength+CPLIDCTopFeedlineWidth/2+CPLIDCTopFeedlineWidth/2 +25+CPLIDCTopFeedlineWidth/2+CPLIDCLength+CPLIDCTopFeedlineWidth/2+IDCtoCLPIDCFeedlineLength+IDCBottomFeedlineWidth)*i+ (DistanceBetweenIDCtoIDC/2+IDCFingerWidth/2+IDCSideShift+IDCFingerWidth+IDCSpacing)) + [0;IDCLength*i] + -u*4*IDCFingerWidth;
    W(u).w = IDCFingerWidth;
end
pixel.W = [pixel.W,W]; %add the wire to list
 
 
%-------------------------------------------------------------------

%                           Picture of Pixel

%-------------------------------------------------------------------
showgeo(pixel);

 
%write to a cif file which can be opened by klayout, xic, LEDIT
%writecifxic(pixel, 'test.cif')