classdef Tanh < dagnn.ElementWise
  methods
    function outputs = forward(obj, inputs, params)
      expPlusPart = exp(2*inputs{1});
      outputs{1} = (expPlusPart-1)./(expPlusPart+1);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      expPlusPart = exp(2*inputs{1});
      expMinusPart = exp(-2*inputs{1});
      derInputs{1} = derOutputs{1}.*4./(expPlusPart+expMinusPart+2);
      derParams = {} ;
    end
  end
end
