% MULTILAYERPERCEPTRON for regression and binary classification 
% This is an implementation of a multi-layer perceptron (MLP), also known
% as a feed-forward artificial neural network.
classdef multiLayerPerceptron < handle
    
    properties
        X       % input data
        y       % output data
        N       % number of data samples
        W       % model weights
        grad    % current gradients
        z       % current hidden variable states
        L       % number of network layers
        h       % activation function
        dh      % derivative of activation function
        J       % loss function
        dJ      % derivative of loss function
        output  % output function
    end
    
    methods
        function obj = multiLayerPerceptron(X, y, graph, activation, mode)
            % MULTILAYERPERCEPTRON assumes graph parameter is given without
            % bias terms
            % initialise data
            obj.X = X;
            obj.y = y;
            [obj.N, ~] = size(X);
            % initialise network
            obj.L = length(graph);
            % append bias units
            for i = 1 : obj.L - 1
                obj.z{i} = [zeros(graph(i), 1) ; 1];
            end
            obj.z{obj.L} = zeros(graph(obj.L), 1);
            % initialise parameters
            for i = 1 : obj.L - 1
                obj.W{i} = rand(graph(i + 1), graph(i) + 1);
                obj.grad{i} = zeros(graph(i + 1), graph(i) + 1);
            end
            % initialise activation function and derivative
            if strcmp(activation, 'tanh')
                obj.h = @(x) tanh(x);
                obj.dh = @(x) 1 - tanh(x).^2;
            elseif strcmp(activation, 'sigma')
                obj.h = @(x) sigma(x);
                obj.dh = @(x) dSigma(x);
            else
                error('Invalid activation function');
            end
            % set cost functions
            if strcmp(mode, 'regression')
                obj.J = @(y, yhat) (y - yhat)^2;
                obj.dJ = @(y, z) -(y - z)';
                obj.output = @(z) z;
            elseif strcmp(mode, 'classification')
                obj.J = @(y, yhat) y * yhat - log(1 + exp(yhat));
                obj.dJ = @(y, z) y - sigma(z);
                obj.output = @(z) sigma(z);
            else
                error('Invalid mode');
            end
        end
        function train(obj, maxIters, alpha, lambda)
            % TRAIN learns model weights with stochastic gradient descent
            iters = 0;
            while iters < maxIters
                cost = 0;
                % randomly permute data
                idx = randperm(obj.N);
                obj.X = obj.X(idx, :);
                obj.y = obj.y(idx, :);
                % run over epoch
                for i = 1 : obj.N
                    obj.forwardPropagate(i);
                    obj.backPropagate(i);
                    obj.updateWeights(alpha, lambda);
                    cost = cost + obj.J(obj.y(i), obj.predict(obj.X(i, :)));
                end
                iters = iters + 1;
                fprintf('Epoch: %d -> Cost: %f\r', [iters, cost / obj.N]);
            end
        end
        function yhat = predict(obj, xhat)
            % PREDICT makes prediction by propagating forwards
            yhat = xhat';
            for i = 1 : obj.L - 1
                yhat = obj.h(obj.W{i} * [yhat; 1]);
            end
            yhat = obj.output(yhat);
        end
    end
    methods(Access = private)
        function forwardPropagate(obj, datum)
            % FORWARDPROPAGATE updates hidden variables with forward
            % propagation
            obj.z{1} = [obj.X(datum, :)' ; 1];
            % update interior layers
            for i = 2 : obj.L - 1
                % include additional unit for bias term
                obj.z{i} = [obj.h(obj.W{i - 1} * obj.z{i - 1}) ; 1];
            end
            obj.z{obj.L} = obj.h(obj.W{obj.L - 1} * obj.z{obj.L - 1});
        end
        function backPropagate(obj, datum)
            % BACKPROPAGATE computes gradients with back propagation
            % final gradient incorporates cost function
            a = obj.W{obj.L - 1} * obj.z{obj.L - 1};
            del = diag(obj.dh(a)) * obj.dJ(obj.y(datum), obj.z{obj.L});
            obj.grad{obj.L - 1} = del * obj.z{obj.L - 1}';
            % update other gradients
            for i = obj.L - 2 : -1 : 1
                a = obj.W{i} * obj.z{i};
                del = diag(obj.dh(a)) * obj.W{i + 1}(:,1:end-1)' * del;
                obj.grad{i} = del * obj.z{i}';
            end
        end
        function updateWeights(obj, alpha, lambda)
            %UPDATEWEIGHTS performs gradient descent step on weights
            for i = 1 : length(obj.W)
                gradient = obj.grad{i} + lambda * obj.W{i};
                obj.W{i} = gradientDescent(alpha, obj.W{i}, gradient);
            end
        end
    end
end
