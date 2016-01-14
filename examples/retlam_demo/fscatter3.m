function [hp] = fscatter3(X,Y,Z,siz,C, cmap, ca)
% [h] = fscatter3(X,Y,Z,siz,C,cmap);
% Plots point cloud data in cmap color classes and 3 Dimensions,
% much faster and very little memory usage compared to scatter3 !
% X,Y,Z,C are vectors of the same length
% X,Y,Z,C might be put in as structure points.x,points.y,points.z,points.int
% with siz the size of each scatter dots
% and C being used as index into colormap (can be any values though)
% cmap is optional colourmap to be used
% h are handles to the line objects

% Felix Morsdorf, Jan 2003 (last update Oct. 2010), Remote Sensing Laboratory Zuerich

% Modified by Neurokernel Developer Team.
 

numclass = max(size(cmap));

% avoid too many calculations

mins = ca(1);
maxs = ca(2);
minz = min(Z);
maxz = max(Z);
minx = min(X);
maxx = max(X);
miny = min(Y);
maxy = max(Y);

% construct colormap :

col = cmap;

% determine index into colormap
C = min(max(C, mins),maxs);
ii = floor( (C - mins ) * (numclass-1) / (maxs - mins) );
ii = ii + 1;

  
hold on
k = 0;o = k;
for j = 1:numclass
  jj = (ii(:)== j);
  if ~isempty(jj)
    k = k + 1;
    h = plot3(X(jj),Y(jj),Z(jj),'o','color',col(j,:),'markersize',siz,'markerfacecolor',col(j,:));
    if ~isempty(h)
      o = o+1;
        hp(o) = h;
    end
  end  
end
caxis(ca)

