%% This visualization code requires MATLAB 2014b or later
% at the end of execution, a movie named "testnat.mp4" will be created.
% The visualization here assumes that you use the default specification,
% i.e., running the simulation by leaving configuration file blank.

%% read simulation results
% read ommatidia coordinates
elevr1 = h5read('retina_elev0.h5','/array');
azimr1 = h5read('retina_azim0.h5','/array');
r1 = 1;
y1 = -r1 .* cos(elevr1) .* sin(azimr1);
x1 = -r1 .* cos(elevr1) .* cos(azimr1);
z1 = r1 .* sin(elevr1);

% read screen coordinates
elevr2 = h5read('grid_dima0.h5','/array');
azimr2 = h5read('grid_dimb0.h5','/array');
r2 = 10;
y = -r2 .* cos(elevr2) .* sin(azimr2);
x = -r2 .* cos(elevr2) .* cos(azimr2);
z = r2 .* sin(elevr2);

% read inputs to and outputs of R1s
inputs = max(h5read('retina_input0.h5','/array'),0);
inputs = inputs(:,1:10:end);
outputs = h5read('retina_output0_gpot.h5','/array');
outputs = outputs(:,1:10:end);

R1input = inputs(1:6:end,:);
R1 = outputs(1:6:end,:);

output_caxis = [min(min(min(outputs(:,501:end)))), max(max(max(outputs(:,501:end))))];
clear('inputs','outputs')
screen = max(h5read('intensities0.h5','/array'),0);

% read L1, L2 response
lam = h5read('lamina_output0_gpot.h5','/array');
lam = lam(:,1:10:end);

total_columnar = 14;

L1 = lam(1:total_columnar:total_columnar*length(elevr1),:);
L2 = lam(2:total_columnar:total_columnar*length(elevr1),:);

% average R1-R6 responses in a cartridge
Rb = (lam(total_columnar-5:total_columnar:total_columnar*length(elevr1),:) + ...
     lam(total_columnar-4:total_columnar:total_columnar*length(elevr1),:) + ...
     lam(total_columnar-3:total_columnar:total_columnar*length(elevr1),:) + ...
     lam(total_columnar-2:total_columnar:total_columnar*length(elevr1),:) + ...
     lam(total_columnar-1:total_columnar:total_columnar*length(elevr1),:) + ...
     lam(total_columnar-0:total_columnar:total_columnar*length(elevr1),:))/6;

clear('lam')
%% visualization

% setup color axis
boxbg = [1,1,1]*0.9569;
weight = 14; % 10
screen_caxis = [0, max(max(max(screen)))];
screen_caxis_gc = [min(log10(screen(:))), max(log10(screen(:)))];
input_caxis = [0, max(max(max(R1input)))];
input_caxis_gc = [min(log10(R1input(:))), max(log10(R1input(:)))];

L1_caxis = [-60,-52];
L2_caxis = [-60,-52];

view1 = [-60, 18];
view2 = [180, 0];

cmap = gray(256);

% set up video writer

fig1 = figure();
set(fig1,'NextPlot','replacechildren');
% set(fig1, 'Color', [0.392, 0.475, 0.635])
set(fig1, 'Color', [1,1,1])

set(fig1,'Position',[50,301, 1280, 1080]);
winsize = get(fig1,'Position');
winsize(1:2) = [0 0];

write_to_file = 1;
if write_to_file
    writerObj = VideoWriter('testnat.mp4', 'MPEG-4');
    writerObj.FrameRate = 20;
    wtiterObj.Quality = 100;
    
    open(writerObj);
end
cmap = gray(256);

% start iterating through frames. The first 0.21 seconds are omitted

for i = 211:10:1000-20
    % screen intensity
    axis1 = subplot('position', [0.05, 0.7, 0.28, 0.28]);
    p1 = surf(x,y,z, screen(:,:,(i-1)/1+1), 'edgecolor','none');
    colormap('gray')
    view(view2)
    caxis(screen_caxis)
    axis equal
    xlim([-10,10])
    ylim([-10,10])
    shading interp
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis1, 'Color', boxbg)
    title('Screen Intensity ', 'FontSize', 16, 'FontWeight', 'bold')

    % inputs to R1
    axis2 = subplot('position', [0.37, 0.7, 0.28, 0.28]);
    h2=fscatter3(x1,y1,z1, weight, R1input(:,i), cmap, input_caxis);
    colormap('gray')
    axis equal
    view(view2)
    caxis(input_caxis)
    xlim([-1,1])
    zlim([-1,1])
    ylim([-1,0])
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis2, 'Color', boxbg)
    title('Inputs (Number of Photons) to R1s ', 'FontSize', 16, 'FontWeight', 'bold')
    
    % 3D perspective
    axis3 = subplot('position', [0.70, 0.7, 0.28, 0.28]);
    p3 = surf(x,y,z, screen(:,:,(i-1)/1+1), 'edgecolor','none');
    colormap('gray')
    view(117,16)
    caxis(screen_caxis)
    axis equal
    xlim([-10,10])
    zlim([-10,10])
    ylim([-10,0])
    shading interp
    grid off
    
    freezeColors
    hold on
    h12=fscatter3(x1,y1,z1, weight/10, R1(:, i), cmap, output_caxis);
    hold off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis3, 'Color', boxbg)
    title('Screen Intensity ', 'FontSize', 16, 'FontWeight', 'bold')
    
    anna3 = annotation(fig1,'textarrow',[0.759375 0.803125],...
    [0.937037037037037 0.882407407407407],'TextEdgeColor','none',...
    'TextLineWidth',1,...
    'FontSize',20,...
    'String',{'screen'},...
    'LineWidth',1);

    anna3a = annotation(fig1,'textarrow',[0.9421875 0.87734375],...
    [0.718518518518519 0.809259259259259],'TextEdgeColor','none',...
    'TextLineWidth',1,...
    'FontSize',20,...
    'String',{'eye'},...
    'LineWidth',1,...
    'Color',[1 0 0]);
    
    % log of Screen intensity
    axis4 = subplot('position', [0.05, 0.375, 0.28, 0.28]);
    p4 = surf(x,y,z, log10(screen(:,:,(i-1)/1+1)), 'edgecolor','none');
    colormap('gray')
    view(view2)
    caxis(screen_caxis_gc)
    axis equal
    xlim([-10,10])
    zlim([-10,10])
    ylim([-10,0])
    shading interp
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis4, 'Color', boxbg)
    title('Log of Screen Intensity', 'FontSize', 16, 'FontWeight', 'bold')

    % Log of R1 inputs
    axis5 = subplot('position', [0.37, 0.375, 0.28, 0.28]);
    h5=fscatter3(x1,y1,z1, weight, log10(R1input(:, i)), cmap, input_caxis_gc);
    colormap('gray')
    axis equal
    view(view2)
    caxis(input_caxis_gc)
    xlim([-1,1])
    zlim([-1,1])
    ylim([-1,0])
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis5, 'Color', boxbg)
    title('Log of Inputs to R1s', 'FontSize', 16, 'FontWeight', 'bold')
    
    % R1 outputs
    axis6 = subplot('position', [0.70, 0.375, 0.28, 0.28]);
    h6=fscatter3(x1,y1,z1, weight, R1(:, i), cmap, output_caxis);
    colormap('gray')
    axis equal
    view(view2)
    caxis(output_caxis)
    xlim([-1,1])
    zlim([-1,1])
    ylim([-1,0])
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis6, 'Color', boxbg)
    title('R1 Response  ', 'FontSize', 16, 'FontWeight', 'bold')
    
    % average R1-R6 response in a cartridge
    axis7 = subplot('position', [0.05, 0.03, 0.28, 0.28]);
    h7 = fscatter3(x1,y1,z1, weight, Rb(:, i), cmap, output_caxis);
    colormap('gray')
    view(view2)
    caxis(output_caxis)
    axis equal
    xlim([-1,1])
    zlim([-1,1])
    ylim([-1,0])
    shading interp
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis7, 'Color', boxbg)
    title({'Average Cartridge ', 'Photoreceptor Terminal Response   '}, 'FontSize', 16, 'FontWeight', 'bold')
    
    % L1 response
    axis8 = subplot('position', [0.37, 0.03, 0.28, 0.28]);
    h8=fscatter3(x1,y1,z1, weight, L1(:, i), cmap, L1_caxis);
    colormap('gray')
    axis equal
    view(view2)
    caxis(L1_caxis)
    xlim([-1,1])
    zlim([-1,1])
    ylim([-1,0])
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis8, 'Color', boxbg)
    title('L1 Response   ', 'FontSize', 16, 'FontWeight', 'bold')
    
    % L2 response
    axis9 = subplot('position', [0.70, 0.03, 0.28, 0.28]);
    h9=fscatter3(x1,y1,z1, weight, L2(:, i), cmap, L2_caxis);
    colormap('gray')
    axis equal
    view(view2)
    caxis(L2_caxis)
    xlim([-1,1])
    zlim([-1,1])
    ylim([-1,0])
    grid off
    xlabel('x', 'FontSize', 16)
    ylabel('y', 'FontSize', 16)
    zlabel('z', 'FontSize', 16)
    set(axis9, 'Color', boxbg)
    title('L2 Response ', 'FontSize', 16, 'FontWeight', 'bold')
    
    % Timer
    anna1 = annotation('textbox', [0.695,0.66,0.1,0.01],...
        'String', [num2str((i-2001)*1), 'ms'], ...
        'FontSize', 16, 'FontWeight', 'Bold', 'LineStyle','none');
    
    drawnow
    if write_to_file
        frame = getframe(fig1,winsize);
        writeVideo(writerObj,frame);
    end
    
    pause(0.001)
    delete(h2)
    delete(h6);
    delete(h5)
    delete(h12);
    delete(h7);
    delete(h8)
    delete(h9);
    delete(anna1);
    delete(anna3);
    delete(anna3a);
end


if write_to_file
    close(writerObj);
end
