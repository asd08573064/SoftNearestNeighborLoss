import torch

class SoftNearestNeighborLoss(nn.Module):
    def __init__(self,
               temperature=100.,
               cos_distance=True):
        super(SoftNearestNeighborLoss, self).__init__()
        
        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = torch.nn.functional.normalize(A, dim=1)
        key_embeddings = torch.nn.functional.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Batched embeddings to compute the SNNL.
            labels: Labels of embeddings.
        """
        batch_size = embeddings.shape[0]
        eps = 1e-9
        
        pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)

        # creating mask to sample same class neighboorhood
        pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        # creating mask to exclude diagonal elements
        ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
        dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

        # all class neighborhood
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        # same class neighborhood
        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

        # adding eps for numerical stability
        # in case of a class having a single occurance in batch
        # the quantity inside log would have been 0
        loss = -torch.log((sacn+eps)/alcn).mean()
        return loss
