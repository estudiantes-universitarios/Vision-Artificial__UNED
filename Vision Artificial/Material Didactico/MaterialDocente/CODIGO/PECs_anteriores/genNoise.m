%% Imagen de entrada normalizada
I = double(imread('FL_s00082_21.png'));
In = (I-min(I(:))) / (max(I(:)-min(I(:)) ) );
size(In)


%% generación de G(x,y)
x = repmat([-256:255] ,424,   1);
y = repmat([-212:211]',  1, 512);
sigma = 50; %150;

P = -( (x).^2 + (y).^2 )   /   (2*sigma^2);
G = (1/2*pi*sigma)* exp(P) ;

imtool(G, [min(G(:)),max(G(:))])  % se selecciona rango de visualización

%g normalizado
Gn = (G-min(G(:))) / (max(G(:)-min(G(:)) ) );

%% R
Rn = rand(size(I));

%% calcular Ir
Ir = In + 0.3*Gn + 0.1*Rn; 
Ir = (Ir-min(Ir(:))) / (max(Ir(:)-min(Ir(:)) ) );
imtool(Ir)


