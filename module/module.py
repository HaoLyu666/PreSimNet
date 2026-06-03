import torch
import torch.nn as nn
class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_reLu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers -2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y

class fun_emb(nn.Module):

    def __init__(self,
                 input_len,
                 enc_dim,
                 num_fun,
                 num_feats,
                 hidden_dim=128,
                 hidden_layers=2):
        super(fun_emb, self).__init__()
        self.input_len = input_len
        self.latent_dim = enc_dim
        self.num_poly = 2
        self.num_sins = self.num_arcsins = num_fun
        self.num_exp =0
        self.hidden_layers = hidden_layers
        self.hidden_dim = self.latent_dim*2
        self.num_feats = num_feats
        self.encoder = MLP(f_in=self.num_feats, f_out=self.num_feats * (self.latent_dim + 2 * self.num_sins + 2 * self.num_arcsins),
                                    activation='relu',
                                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)

    def forward(self, inps):
        # the encoder learns the coefficients of basis functions
        encoder_outs = self.encoder(inps)  # b sql feat*dim

        # reshape inputs and encoder outputs for next step
        encoder_outs = encoder_outs.reshape(inps.shape[0], inps.shape[1],
                                            (self.latent_dim + self.num_sins * 2 + 2 * self.num_arcsins),
                                            self.num_feats)  # b sql dim feat

        # the input to the measurement functions are
        # the muliplication of coeffcients and original observations.
        coefs = torch.einsum("bskd, bsd -> bsk", encoder_outs, inps)  # b sql dim
        #####################################################

        ################ Calculate Meausurements ############
        embedding = torch.zeros(encoder_outs.shape[0], encoder_outs.shape[1],
                                self.latent_dim).to(inps.device)  # b sql dim2
        for f in range(encoder_outs.shape[1]):
            # polynomials
            for i in range(self.num_poly):
                embedding[:,  f, i] = coefs[:, f, i] ** (i+1)

            # # exponential function
            # for i in range(self.num_poly, self.num_poly + self.num_exp):
            #     embedding[:,  f, i] = torch.exp(coefs[:,  f, i])

            # sine/cos functions
            for i in range(self.num_poly + self.num_exp,
                           self.num_poly + self.num_exp + self.num_sins):
                embedding[:,  f,
                i] = coefs[:,  f, self.num_sins * 2 + i] * torch.cos(
                    coefs[:,  f, i])
                embedding[:,  f, self.num_sins +
                                   i] = coefs[:,  f, self.num_sins * 3 + i] * torch.sin(
                    coefs[:,  f, self.num_sins + i])
            # arcsine/arccos functions
            for i in range(self.num_poly + self.num_exp + self.num_sins*2,
                           self.num_poly + self.num_exp + self.num_sins*2 + self.num_arcsins):
                embedding[:, f,
                i] = coefs[:, f, self.num_arcsins * 2 + i] * torch.cos(
                    coefs[:, f, i])
                embedding[:, f, self.num_arcsins +
                                i] = coefs[:, f, self.num_arcsins * 3 + i] * torch.sin(
                    coefs[:, f, self.num_arcsins + i])
            # # exponential function
            # for i in range(self.num_poly + self.num_exp + self.num_sins*2 + self.num_arcsins*2, self.num_poly + self.num_exp + self.num_sins*2 + self.num_arcsins*2 + self.num_log):
            #     embedding[:, f, i] = torch.log(coefs[:, f, i])

            # the remaining ouputs are purely data-driven measurement functions.
            embedding[:,  f, self.num_poly + self.num_exp + self.num_sins *2 + self.num_arcsins*2:] = coefs[:,  f, self.num_poly + self.num_exp + self.num_sins *4 + self.num_arcsins*4:]
        return embedding


class BERTTimeEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim):
        super(BERTTimeEmbedding, self).__init__()
        self.embeddings = nn.Embedding(max_position_embeddings, embedding_dim)

    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(*(input_ids.shape[0], -1))
        return self.embeddings(position_ids)