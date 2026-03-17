# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.


### Dataset Name: ner_dataset.csv

Size: Contains thousands of words grouped into sentences with entity annotations.

#### Columns:

Sentence # – Sentence ID

Word – Individual word/token in the sentence

POS – Part-of-speech tag

Tag – Named entity tag (e.g., O, B-PER, I-LOC, etc.)
## Design Steps:

### STEP 2
Load the NER dataset and fill missing values.
### STEP 3
Create word and tag dictionaries for encoding.
### STEP 4
Group words into sentences and encode them into numbers.
### STEP 5
Build a BiLSTM model for sequence tagging.
### STEP 6
Train the model using the training data.
### STEP 7
Evaluate the model performance on test data.


## PROGRAM
### Name:Dhivya Dharshini B
### Register Number:212223240031
```python
class BiLSTMTagger(nn.Module):
    # Include your code here
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100): # Changed _init_ to __init__
        super(BiLSTMTagger, self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.dropout=nn.Dropout(0.1)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_dim*2,tagset_size)


    def forward(self, input_ids):
        # Include your code here
        x = self.embedding(input_ids) # Changed x to input_ids
        x = self.dropout(x)
        x,_ = self.lstm(x)
        return self.fc(x)


     

model=BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
     


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_loss = 0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          val_loss += loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

    return train_losses, val_losses
     

def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != tag2idx["O"]:
                        true_tags.append(idx2tag[labels[i][j].item()])
evaluate_model(model, test_loader, X_test, y_test)
print('Name: Dhivya Dharshini B')
print('Register Number: 212223240031')
history_df = pd.DataFrame({"loss": train_losses, "val_loss": val_losses})
history_df.plot(title="Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="688" height="548" alt="image" src="https://github.com/user-attachments/assets/fe974e2a-8161-45a8-b90e-9de5654cc38e" />


### Sample Text Prediction
<img width="382" height="474" alt="image" src="https://github.com/user-attachments/assets/60897bcf-c0ab-4e97-b9c4-30ab9b9de7bb" />



## RESULT
The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.
