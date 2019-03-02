function h = stacked_bar3(array)
    if any(array(:) < 0)
        error('Only positive values supported')
    end
    
    dims = size(array);
    if any(dims==0)
        error('Empty dimensions are not supported')
    end    

    switch length(dims)
        case 2
            ns = 1;
        case 3
            ns = dims(3);
        otherwise
            error('Must be a 3D array')
    end
    nr = dims(1);
    nc = dims(2);
    
    ax = newplot;
    co = ax.ColorOrder;    
    h = gobjects(1,ns);
    view(ax,3)
    xlim(ax,[.5 nc+.5])
    ylim(ax,[.5 nr+.5])
    
    bw = .4;
    offmat = [-bw, +bw, 0; ...
              -bw, -bw, 0; ...
              +bw, -bw, 0; ...
              +bw, +bw, 0];
    sidemat = [1, 2, 2, 1; ...
               2, 3, 3, 2; ...
               3, 4, 4, 3; ...
               4, 1, 1, 4] ...
            + repmat([0, 0, 4*nr*nc, 4*nr*nc],[4, 1]);
    topmat = (1:4) + 4*nr*nc;

    top = zeros(dims(1:2));
    for s = 1:ns
        bottom = top;
        top = bottom + array(:,:,s);

        verts = zeros(4*nr*nc*2, 3);
        faces = ones(5*nr*nc, 4);
        for r = 1:nr
            for c = 1:nc
                vindex = 4*(r-1 + nr*(c-1));
                lindex = 5*(r-1 + nr*(c-1));
                rindex = 4*(r-1 + nr*(c-1));
                verts(vindex +           (1:4)', :) = repmat([c,r,bottom(r,c)],[4,1]) + offmat;
                verts(vindex + 4*nr*nc + (1:4)', :) = repmat([c,r,   top(r,c)],[4,1]) + offmat;
                faces(lindex + (1:5)',:) = rindex + [sidemat; topmat];
            end
        end
        
        cix = 1+mod(s-1, size(co,1));
        h(s) = patch('Vertices', verts, ...
                     'Faces', faces, ...
                     'FaceColor', co(cix,:), ...
                     'Parent', ax);
                 
        bottom = top;
    end
end