import express from 'express';
import { setEmailRoutes } from './routes/emailRoutes';
import { logger } from './utils/logger';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
setEmailRoutes(app);

// Start the server
app.listen(PORT, () => {
    logger.info(`Server is running on port ${PORT}`);
});