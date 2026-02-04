function z = PnPEIT(J,u,varargin)
%% 
%  PnPEIT -> sparse and smooth Electrical Impedance Tomography (EIT) 
%            image reconstruction via Plug-and-play ADMM 
%
%% --------------- Description --------------------------------------------
%
%  PnPEIT solves the following Graph Laplacian-l1 optimization  problem 
%  [size(J) = (M,N); size(s) = (N,1)]; size(u) = (M,1)]
%
%         min  (1/2) ||J s-u||^2_2 + alpha ||s||_1+beta/2 s'Ls
%          s                      
%
%% -------------------- Line of Attack  -----------------------------------
%
%  PnPEIT solves the above optimization problem by 
%  the augmented Lagrangian method of multipliers (ADMM) . 
% 
% 
%         min  (1/2) ||J s-u||^2_2 + alpha ||z||_1 + beta/2 s'Ls
%         s,z              
%         subject to:   s = z
%
%  Augmented Lagrangian (scaled version):
%
%       L(s, z, d) = (1/2) ||J s-u||^2_2 + alpha ||z||_1 + beta/2 s'Ls 
%                 + d'(s-z)+ mu/2||s-z||^2_2
%       
%  where d is the dual variable.
%
%
%  ADMM:
%
%      do 
%        X  <-- arg min L(s,z,d)
%                    x 
%        Z  <-- arg min L(s,z,d)
%                    z
%        d  <-- d - mu(s-z);
%      while ~stop_rulde
%  
% For details see
%
%
% [1] Shuaikai Shi, Ruiyuan Kang and Panos Liatsis, "Fast Electrical Impedance
% Tomography with Hybrid Priors", 2026
%
%
% ------------------------------------------------------------------------
%%  ===== Required inputs =============
%
%  J - [M(measurement) x N(elements)] Jacobian matrix
%
%  u - matrix with  M(measurement) x 1.
%
%      
%
%
%%  ====================== Optional inputs =============================
%
%  'AL_ITERS' - Minimum number of augmented Lagrangian iterations
%               Default: 100;
%               
%  alpha1 - regularization parameter for sparseness. alpha is a scalar
%           Default: 0. 
%  beta1 - regularization parameter for smoothness. beta1 is a scalar
%           Default: 0. 
%
%
% 
%   'TOL'    - tolerance for the primal and  dual residuals 
%              Default = 1e-4; 
%
%
%  'verbose'   = {'yes', 'no'}; 
%                 'no' - work silently
%                 'yes' - display warnings
%                  Default 'no'
%        
%%  =========================== Outputs ==================================
%
% s  =  [Nx1] estimated conductivity matrix
%
%

%%
% ------------------------------------------------------------------
% Author: Shuaikai Shi, 2026
%
%
%
%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end
% mixing matrixsize
[LM,N] = size(J);
% data set size
[L,p] = size(u);
if (LM ~= L)
    error('Jacobian matrix J and Voltages u are inconsistent');
end

%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
% maximum number of AL iteration
AL_iters = 100;
% regularizatio parameter
alpha1 = 0;
% display only sunsal warnings
verbose = 'off';
% tolerance for the primal and dual residues
tol = 1e-4;


%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'AL_ITERS'
                AL_iters = round(varargin{i+1});
                if (AL_iters <= 0 )
                       error('AL_iters must a positive integer');
                end
            case 'ALPHA1'
                alpha1 = varargin{i+1};
                if (sum(sum(alpha1 < 0)) >  0 )
                       error('alpha must be positive');
                end
            case 'TOL'
                tol = varargin{i+1};
            case 'VERBOSE'
                verbose = varargin{i+1};
            case 'BETA1'
                beta1 = varargin{i+1};

            case 'IND'
               ind = varargin{i+1} ;

            case 'L'
                L = varargin{i+1};
            case 'WIDTH'
                width = varargin{i+1};
            case 'METHOD'
                method = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%%
%---------------------------------------------
% just least squares
%---------------------------------------------
if sum(sum(alpha1 == 0)) &&  (beta1 == 0)
    z = pinv(J)*u;
    return
end



%%
%---------------------------------------------
%  Constants and initializations
%---------------------------------------------
mu_AL = 0.01;
mu = 10*alpha1 + mu_AL;

[UF,SF] = svd(J'*J);
sF = diag(SF);
IF = UF*diag(1./(sF+mu))*UF';


%%
%---------------------------------------------
%  Initializations
%---------------------------------------------
x= IF*J'*u;
z = x;

% scaled Lagrange Multipliers
d  = 0*z;


%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
tol1 = sqrt(N*p)*tol;
tol2 = sqrt(N*p)*tol;
i=1;
res_p = inf;
res_d = inf;
mu_changed = 0;


 
 
 

 
 
% PnP restoration parameters
if strcmp(method, 'SL-NLM')
    searchRad =40;      % NLM search window radius
    patchRad = 5;       % NLM patch radius
    h = 0.2; 
    W_warm = @(x) JNLM(x,x,patchRad,searchRad,h);
end
IF=pinv(J' *J+ mu* eye(N)+beta1*L);
Ju=J'*u;
while (i <= AL_iters) && ((abs (res_p) > tol1) || (abs (res_d) > tol2)) 
    % save z to be used later
    if mod(i,10) == 1
        z0 = z;
    end
   
    %% update x
    x=IF*(Ju+mu*(z+d));%

 
    % minimize with respect to z
    %% update z
    z =  soft(x-d,alpha1/mu);
    
    %% plug and play
    pad=zeros(width,width);
    pad=pad(:);
    pad(ind)=z;
    pad=reshape(pad,[width,width]);

    pad_min=min(pad(:));
    scale = max(pad(:))  - min(pad(:))+1e-12;
    pad = (pad - pad_min) / scale;
    %% BM3D
    window_size = 8;
    [h_old, w_old] = size(pad);
    
    h_pad = (floor(h_old / window_size) + 1) * window_size - h_old;
    w_pad = (floor(w_old / window_size) + 1) * window_size - w_old;
    
    %   pad h
    pad = cat(1, pad, flip(pad, 1));
    pad = pad(1:h_old + h_pad, :, :, :);
    
    %    pad w
    pad = cat(2, pad, flip(pad, 2));
    pad = pad(:, 1:w_old + w_pad, :, :);
    
     
    % BM3D 
    if strcmp(method, 'GLSL-BM3D')
        pad = BM3D_matlab(pad,10);
    end
    % NLM
    if strcmp(method, 'SL-NLM')
        pad=W_warm(pad);
    end
    
    % clip
    pad = pad(1:h_old, 1:w_old, :, :);
    pad = (pad  * scale + pad_min);
    
    pad=pad(:);
    z=pad(ind);
    %% end plug-and-play
   
 
 
     
 
    %% Lagrange multipliers update (dual variable)
    d = d -(x-z);

    % update mu so to keep primal and dual residuals whithin a factor of 10
    if mod(i,10) == 1
        % primal residue
        res_p = norm(x-z,'fro');
        % dual residue
        res_d = mu*norm(z-z0,'fro');
        if  strcmp(verbose,'yes')
            fprintf(' i = %f, res_p = %f, res_d = %f\n',i,res_p,res_d)
        end
        % update mu
        if res_p > 10*res_d
            mu = mu*2;
            d = d/2;
            mu_changed = 1;
        elseif res_d > 10*res_p
            mu = mu/2;
            d = d*2;
            mu_changed = 1;
        end
        if  mu_changed
            % update IF  

            IF=pinv(J' *J+ mu* eye(N)+beta1*L);
            mu_changed = 0;
            %mu
        end   
    end
    i=i+1;
    % res_pp(i) = norm(x-z,'fro');
end
end

    
 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function y = soft(x,T)
    T = T + eps;
    y = max(abs(x) - T, 0);
    y = y./(y+T) .* x;
end