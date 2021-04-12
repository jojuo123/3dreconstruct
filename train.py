import torch 
import data_loader
import reconstruction_model
import options
import torch.optim as optim


def train():

    opt = options.Option()
    
    dataloader = data_loader.data_loader()
    net = reconstruction_model.FullModel(opt)

    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    for epoch in range(opt.train_maxiter):
        running_loss = 0.0
        total_mini_batch = 0
        for idx, data in enumerate(dataloader):
            total_mini_batch += 1
            inputs = data
            optimizer.zero_grad()

            loss = net(input)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('%d loss: %.5f' % (epoch + 1, running_loss / total_mini_batch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'loss': loss
        }, opt.save_path)
