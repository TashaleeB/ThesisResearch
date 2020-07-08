function showgeo(obj)
if(isfield(obj,'P'))
    for u=1:numel(obj.P)
        z = obj.P(u).z;
        [f, v] = poly2fv(real(z), imag(z));
        patch('Faces', f, 'Vertices', v, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha',1);        
        axis image
        hold on;
    end
end

if(isfield(obj,'W'))
    for u=1:numel(obj.W)
        wire2poly(obj.W(u).z, obj.W(u).w);
    end
end

function wire2poly(z, w)
d = w/2;
za = z;zb =z;
for u = 1:numel(z)
    if(u==1)
        e1 = (z(2) - z(1))/abs(z(2) - z(1));
        za(1) = z(1) + e1*i * d;
        zb(1) = z(1) - e1*i * d;
        
    elseif(u<numel(z))
        e1 = (z(u) - z(u-1))/abs(z(u) - z(u-1));
        e2 = (z(u) - z(u+1))/abs(z(u) - z(u+1));
        if(abs(e1+e2)<1e-6)%straight
            e3 = e1*i * d;
        else
            l = d/sin(angle(e2/e1));
            e3 = l*e1 + l*e2;
        end
        za(u) = z(u) + e3;
        zb(u) = z(u) - e3;
    else
        e1 = (z(u) - z(u-1))/abs(z(u) - z(u-1));
        za(u) = z(u) + e1*i * d;
        zb(u) = z(u) - e1*i * d;
    end
    
end
zc = [za;zb(end:-1:1);za(1)];


[f, v] = poly2fv(real(zc), imag(zc));
patch('Faces', f, 'Vertices', v, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha',1);
axis image
hold on;
plot(z+1e-10i,'k-')