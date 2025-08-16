
import unittest
from models.user import User
import os

class TestUser(unittest.TestCase):
    def test_user_creation(self):
        user = User(id=1, username='testuser', email='test@example.com')
        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.email, 'test@example.com')
        self.assertTrue(user.is_active)

    def test_user_inactive(self):
        user = User(id=2, username='inactive', email='inactive@example.com', is_active=False)
        self.assertFalse(user.is_active)

    def test_deepface_gender_detection(self):
        try:
            from deepface import DeepFace
            # Use a sample image path (replace with a real image path for actual test)
            sample_image = os.path.join(os.path.dirname(__file__), 'sample1.jpg')
            if not os.path.exists(sample_image):
                self.skipTest('Sample image not found for DeepFace test.')
            result = DeepFace.analyze(img_path=sample_image, actions=['gender'], enforce_detection=False)
            self.assertIn('gender', result if isinstance(result, dict) else result[0])
        except ImportError:
            self.skipTest('DeepFace not installed.')
        except Exception as e:
            self.fail(f'DeepFace gender detection failed: {e}')

if __name__ == '__main__':
    unittest.main()
