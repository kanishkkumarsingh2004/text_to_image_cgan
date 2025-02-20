from data import epochs, dataloader, dataset, torch, device, noise_dim, optimizer_D, optimizer_G, disc, gen, criterion, save_folder, os, nn

# Remove spectral normalization from model weights before saving
def remove_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            try:
                # Remove spectral norm and keep the original weight
                nn.utils.remove_spectral_norm(module)
            except:
                pass

# Training Loop
for _epoch in range(epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        # Clear GPU cache at the start of each batch
        torch.cuda.empty_cache()
        
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        noise = torch.randn(batch_size, noise_dim, device=device)
        # Convert numeric labels to class names
        real_labels = [str(dataset.classes[label]) for label in labels.tolist()]
        # Generate fake labels using class names
        fake_labels = [str(dataset.classes[i]) for i in torch.randint(0, len(dataset.classes), (batch_size,)).tolist()]

        real_targets = torch.ones(batch_size, device=device)
        fake_targets = torch.zeros(batch_size, device=device)

        ####### Train Discriminator #######
        optimizer_D.zero_grad()

        # Reshape images to match discriminator input dimensions (batch_size, channels*height*width)
        real_images_flat = real_images.view(batch_size, -1)
        real_outputs = disc(real_images_flat, real_labels)
        d_loss_real = criterion(real_outputs, real_targets)
        d_loss_real.backward()

        fake_images = gen(noise, fake_labels)
        # Reshape fake images to match discriminator input dimensions
        fake_images_flat = fake_images.view(batch_size, -1)
        fake_outputs = disc(fake_images_flat.detach(), fake_labels)
        d_loss_fake = criterion(fake_outputs, fake_targets)
        d_loss_fake.backward()

        optimizer_D.step()

        ####### Train Generator #######
        optimizer_G.zero_grad()

        # Re-generate fake images for generator training
        fake_images = gen(noise, fake_labels)
        fake_images_flat = fake_images.view(batch_size, -1)
        fake_outputs = disc(fake_images_flat, fake_labels)
        g_loss = criterion(fake_outputs, real_targets)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{_epoch+1}/{epochs}], D Loss: {(d_loss_real.item() + d_loss_fake.item()):.4f}, G Loss: {g_loss.item():.4f}")

    # Clear GPU cache every batch
    torch.cuda.empty_cache()

    os.makedirs(save_folder, exist_ok=True)
    
    # Remove spectral norm before saving
    remove_spectral_norm(gen)
    remove_spectral_norm(disc)
    
    torch.save(gen.state_dict(), os.path.join(save_folder, "Generator.pth"))
    torch.save(disc.state_dict(), os.path.join(save_folder, "Discriminator.pth"))