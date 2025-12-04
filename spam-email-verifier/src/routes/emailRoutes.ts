import { Router } from 'express';
import EmailController from '../controllers/emailController';

const router = Router();
const emailController = new EmailController();

export default function setEmailRoutes(app) {
    app.use('/api/emails', router);
    router.post('/verify', emailController.verifyEmail.bind(emailController));
    router.get('/status/:id', emailController.getVerificationStatus.bind(emailController));
}