const Arweave = require('arweave/node')
const fs = require('fs');

const arweave = Arweave.init({
    host: 'arweave.net',
    port: 443,
    protocol: 'https',
    timeout: 20000,
    logging: false,
})

const jwk = JSON.parse(fs.readFileSync('./arweave-keyfile.json', 'utf8'))

module.exports = {

    sendData: async function (randomFeed) {
        let transaction = await arweave.createTransaction({ data: randomFeed }, jwk);
        transaction.addTag('title', 'Speech to Text to Blockchain');
        transaction.addTag('app-name', 'Machine Learning Feeds');

        await arweave.transactions.sign(transaction, jwk);
        await arweave.transactions.post(transaction);
    }
}