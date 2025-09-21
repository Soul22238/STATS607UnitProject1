# Run this lines to reload the model
bert_model = DistilBertModel.from_pretrained(model_name)
model = DistilBertForTokenClassification(bert_model, hidden_dim, num_pos_tags)

# Run some test optimization
model.to(device)
optim = get_optimizer(model, lr=5e-5, weight_decay=0)
best_model, stats = train_model(model, val_loader, val_loader, optim,
                                num_epoch=25, collect_cycle=5, device=device)

plot_loss(stats)

# Trian the full model
# Run this lines to reload the model
bert_model = DistilBertModel.from_pretrained(model_name)
model = DistilBertForTokenClassification(bert_model, 768, num_pos_tags)

# Run the full optimization
model.to(device)
optim = get_optimizer(model, lr=5e-5, weight_decay=0)
best_model, stats = train_model(model, train_loader, val_loader, optim,
                                num_epoch=10, collect_cycle=20, device=device)