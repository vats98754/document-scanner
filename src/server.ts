import express from 'express';
import path from 'path';

const app = express();
const PORT: number = parseInt(process.env.PORT || '8080', 10);

// Serve static files from the current directory
app.use(express.static(path.join(__dirname, '..')));

// Route for the main page
app.get('/', (req: express.Request, res: express.Response) => {
    res.sendFile(path.join(__dirname, '..', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`🚀 Document Scanner app running at http://localhost:${PORT}`);
    console.log('📷 Make sure to allow camera permissions when prompted');
});
