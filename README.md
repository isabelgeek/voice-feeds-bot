# VOICE FEEDS BOT

## Introduction

This is an Alexa skill that provided a list of interesting feeds about a topic, the Alexa assistant will select a feed at random, store the Arwewave's blockchain network and tell it to the user when the skill is invoked.

In my opinion to evaluate data streams is important to have in mind the quality and utility of the data and the audience it reaches. In this case the bot / assistant notarizes eaach data stream that it tells to the users.

This is the link of the Alexa skill live in Amazon marketplace for skills:
https://amzn.to/36yQ7jD

Right now includes different languages locales such as en-AU, en-CA, en-GB, en-IN, and en-US and it is published in the following countries: United States, United Kingdom, Canada, Australia and India.
I will add in the near future more locales such Spanish for Mexico and Spain.

More topics will be added and published in additional skills.

## Wallet address archiving data stream - feeds
This is the wallet address used to store the feeds in the project:
YMXJ-1WWiTnR2G6sC-wOZQPfwjyB-Lk1-hJxqAKlpHs

I have also used the tags:
`title`= _Speech to Text to Blockchain_
`app-name` = _Machine Learning Feeds_



## Installation

### Create the Voice User Interface

1. Go to the [Amazon Developer Portal](http://developer.amazon.com "Amazon Developer Portal).

2. Once you have signed in, go to the **Alexa Console** and Select the **Skills** link.

3. From the **Alexa Skills Console** select the **Create Skill** button and give your new skill a **Name**.
This is the name that will be shown in the Amazon Alexa Skills section.

4. Select the **Custom** model button to add it to your skill, and select the **Create Skill** button now at the top right.

5. **Build the Interaction Model for your skill**.
On the left hand navigation panel, select the **JSON Editor** tab under **Interaction Model**.
Then replace any existing code with the code provided in the folder `models` of this repo and then **Save Model**.

Select the **Invocation** tab and fill in the skill invocation name. 
This is the name the users will need to say to start the skill. Finally, click "Build Model".


### Setting Up a Lambda function Using Amazon Web Services
You will need to create a Lambda function in the AWS developer console.
1. You will be creating an AWS Lambda function using [Amazon Web Services](http://aws.amazon.com "Amazon Web Services").

2. Once you find Lambda in the list of services, **check the AWS region**. 
Please note that AWS Lambda only works with the Alexa Skills Kit in some regions: US East (N. Virginia), US West (Oregon), Asia Pacific (Tokyo)  and EU (Ireland).

3. Click the "Create function" button and create the Lambda function.
Remember to grant the Alexa Skills Kit permission to invoke it, and set up the IAM role. 

It's time that you install the dependencies of the repo you have cloned. 
You will need Node.js (> v8).

Go to the _custom_ folder which is inside the _lambda_ folder and introduce the following command in the CLI:
`npm install`

Now upload a zip of all the content within the custom folder (including the node_modules).

4. Copy the Amazon Resource Name (ARN) for this function as you will need later.


### Connecting your voice user interface to your Lambda function.

1. Return back to the Amazon Developer Portal.
While on the **Build** tab, select the **Endpoint** tab.
**Select the "AWS Lambda ARN" option** for your **endpoint**. 
Introduce the ARN you have copied before and save the endpoint.


## Testing 

Go back to the [Amazon Developer Portal](https://developer.amazon.com "Amazon Developer Portal") and access the **Alexa Simulator**, by selecting the **Test** tab from the navigation menu. 
Then you can invoke the skill to validate that the skill is working as expected.


## Customize the Skill

In order to make it your own, you will need to customize it with data and responses that you create.

You will neee to provide a set of feeds for your topic replacing the feeds in `index.js` and new sentences to respond to your users.

## Get Your Skill Certified and Published

Please follow the instructions and guidelines from Amazon to certify and get published your skill.

## License
The source code is licensed under Apache License Version 2.0
[LICENSE](https://github.com/isabelgeek/voice-feeds-bot/blob/master/LICENSE "LICENSE")

The machine learning feeds and vocabulary used in this bot is licensed by Google under the [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/ "Creative Commons Atribution 4.0 License") by Google

Arweave, Google and Alexa are trademarks of their respective owners.


