function [w, state] = adam(w, state, grad, opts, lr)
%ADAM
%   Example Adam.
% Copyright (C) 2016 Sung-Ho Bae
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 0 % Return the default solver options
  w = struct('beta1', 0.5, 'beta2', 0.999, 'epsilon', 1e-8);
  return;
end


if isempty(state)
  state.m = single(0);
  state.v = single(0);
  state.t = single(0);
end

beta1 = opts.beta1;
beta2 = opts.beta2;
epsilon = opts.epsilon ;

grad = gather(grad);

state.m = beta1.*state.m +(1-beta1).*grad;
state.v = beta2.*state.v +(1-beta2).*grad.*grad;
state.t = state.t + 1;

m_hat = state.m./(1-beta1.^state.t);
v_hat = state.v./(1-beta2.^state.t);

w = w -lr.*m_hat./( sqrt(v_hat)+epsilon);
