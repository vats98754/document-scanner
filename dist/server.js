"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const path_1 = __importDefault(require("path"));
const app = (0, express_1.default)();
const PORT = parseInt(process.env.PORT || '8080', 10);
// Serve static files from the current directory
app.use(express_1.default.static(path_1.default.join(__dirname, '..')));
// Route for the main page
app.get('/', (req, res) => {
    res.sendFile(path_1.default.join(__dirname, '..', 'index.html'));
});
app.listen(PORT, () => {
    console.log(`🚀 Document Scanner app running at http://localhost:${PORT}`);
    console.log('📷 Make sure to allow camera permissions when prompted');
});
//# sourceMappingURL=server.js.map