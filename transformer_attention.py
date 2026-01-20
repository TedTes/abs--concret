from manim import *
import numpy as np

class CompleteTransformerPipeline(Scene):
    def construct(self):
        # Configuration
        self.sentence = "the cat sat on the mat"
        self.tokens = ["the", "cat", "sat", "on", "the", "mat"]
        self.n_tokens = 6
        self.d_model = 4  # Embedding dimension (simplified)
        self.d_k = 2  # Query/Key dimension
        
        # Define colors (matching your document)
        self.COLOR_TEXT = BLACK
        self.COLOR_EMBEDDING = BLUE
        self.COLOR_PE = PURPLE
        self.COLOR_Q = ORANGE
        self.COLOR_K = ORANGE
        self.COLOR_V = ORANGE
        self.COLOR_SCORES = RED
        self.COLOR_WEIGHTS = GREEN
        self.COLOR_OUTPUT = TEAL
        self.COLOR_FINAL = "#1E3A8A"  # Deep blue
        
        # Initialize matrices with realistic values
        np.random.seed(42)
        self.E = np.random.randn(self.n_tokens, self.d_model) * 0.5 + 0.5  # Embeddings
        self.E = np.clip(self.E, 0, 1)  # Keep values reasonable
        
        # Positional encoding (simplified sine/cosine)
        self.P = np.zeros((self.n_tokens, self.d_model))
        for pos in range(self.n_tokens):
            for i in range(self.d_model):
                if i % 2 == 0:
                    self.P[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                else:
                    self.P[pos, i] = np.cos(pos / (10000 ** ((i-1) / self.d_model)))
        
        self.X = self.E + self.P  # Combined input
        
        # Weight matrices
        self.WQ = np.random.randn(self.d_model, self.d_k) * 0.5
        self.WK = np.random.randn(self.d_model, self.d_k) * 0.5
        self.WV = np.random.randn(self.d_model, self.d_k) * 0.5
        
        # Compute Q, K, V
        self.Q = self.X @ self.WQ
        self.K = self.X @ self.WK
        self.V = self.X @ self.WV
        
        # Attention computation
        self.scores = (self.Q @ self.K.T) / np.sqrt(self.d_k)
        self.attention_weights = self.softmax(self.scores)
        self.Z = self.attention_weights @ self.V
        
        # Run complete pipeline
        self.scene_1_raw_text()
        self.wait(1)
        self.scene_2_tokenization()
        self.wait(1)
        self.scene_3_embeddings()
        self.wait(1)
        self.scene_4_positional_encoding()
        self.wait(1)
        self.scene_5_qkv_projections()
        self.wait(1)
        self.scene_6_attention_scores()
        self.wait(1)
        self.scene_7_softmax()
        self.wait(1)
        self.scene_8_value_mixing()
        self.wait(1)
        self.scene_9_semantic_emergence()
        self.wait(2)
    
    def softmax(self, x):
        """Compute softmax along rows"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def create_matrix_visual(self, matrix, color=WHITE, cell_size=0.5, scale=1.0, show_values=True):
        """Create a visual matrix using colored cells"""
        rows, cols = matrix.shape
        matrix_group = VGroup()
        
        for i in range(rows):
            row_group = VGroup()
            for j in range(cols):
                cell = Square(side_length=cell_size, stroke_color=color, stroke_width=2, fill_opacity=0.1, fill_color=color)
                if show_values:
                    value_text = Text(f"{matrix[i,j]:.2f}", font_size=12, color=color)
                    value_text.move_to(cell.get_center())
                    cell_group = VGroup(cell, value_text)
                else:
                    cell_group = cell
                row_group.add(cell_group)
            row_group.arrange(RIGHT, buff=0)
            matrix_group.add(row_group)
        
        matrix_group.arrange(DOWN, buff=0)
        matrix_group.scale(scale)
        return matrix_group
    
    def scene_1_raw_text(self):
        """Scene 1: Raw text input"""
        title = Text("Scene 1: Input Text", font_size=36, color=self.COLOR_TEXT, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show raw sentence
        sentence = Text(f'"{self.sentence}"', font_size=32, color=self.COLOR_TEXT)
        sentence.move_to(ORIGIN)
        self.play(Write(sentence), run_time=2)
        
        self.wait(1)
        self.current_title = title
        self.sentence_visual = sentence
    
    def scene_2_tokenization(self):
        """Scene 2: Tokenization"""
        new_title = Text("Scene 2: Tokenization", font_size=36, color=self.COLOR_TEXT, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        # Create token boxes
        token_boxes = VGroup()
        for i, token in enumerate(self.tokens):
            box = Rectangle(width=1.2, height=0.7, color=self.COLOR_TEXT, stroke_width=3)
            text = Text(token, font_size=18, color=self.COLOR_TEXT)
            text.move_to(box.get_center())
            # Add token index
            index = Text(f"t{i+1}", font_size=12, color=GRAY).next_to(box, DOWN, buff=0.1)
            token_group = VGroup(box, text, index)
            token_boxes.add(token_group)
        
        token_boxes.arrange(RIGHT, buff=0.2)
        token_boxes.move_to(ORIGIN)
        
        # Transform sentence to tokens
        self.play(Transform(self.sentence_visual, token_boxes), run_time=2)
        
        # Show formula
        formula = Text("tokens → [t₁, t₂, t₃, t₄, t₅, t₆]", font_size=20)
        formula.next_to(token_boxes, DOWN, buff=0.8)
        self.play(FadeIn(formula))
        
        self.wait(1)
        self.tokens_visual = self.sentence_visual
        self.formula = formula
    
    def scene_3_embeddings(self):
        """Scene 3: Token embeddings"""
        new_title = Text("Scene 3: Embedding Matrix (X)", font_size=36, color=self.COLOR_EMBEDDING, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        self.play(FadeOut(self.formula))
        
        # Create embedding matrix
        E_matrix = self.create_matrix_visual(self.E, color=self.COLOR_EMBEDDING, scale=1.0)
        E_matrix.move_to(ORIGIN + UP * 0.3)
        
        # Transform tokens to matrix
        self.play(Transform(self.tokens_visual, E_matrix), run_time=2)
        
        # Show formula
        formula = Text("X ∈ ℝ⁶ˣ⁴", font_size=28, color=self.COLOR_EMBEDDING)
        formula.next_to(E_matrix, DOWN, buff=0.6)
        self.play(FadeIn(formula))
        
        explanation = Text("Each token → d-dimensional vector", font_size=18, color=GRAY)
        explanation.next_to(formula, DOWN, buff=0.3)
        self.play(FadeIn(explanation))
        
        self.wait(1)
        self.E_visual = self.tokens_visual
        self.formula = formula
        self.explanation = explanation
    
    def scene_4_positional_encoding(self):
        """Scene 4: Add positional encoding"""
        new_title = Text("Scene 4: Positional Encoding (X + P)", font_size=36, color=self.COLOR_PE, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        self.play(FadeOut(self.formula), FadeOut(self.explanation))
        
        # Show positional encoding matrix
        P_matrix = self.create_matrix_visual(self.P, color=self.COLOR_PE, scale=0.8)
        P_label = Text("P (Position)", font_size=20, color=self.COLOR_PE)
        P_group = VGroup(P_label, P_matrix).arrange(DOWN, buff=0.2)
        P_group.move_to(LEFT * 3.5)
        
        # Show combined matrix
        X_matrix = self.create_matrix_visual(self.X, color=BLUE, scale=0.8)
        X_label = Text("X = E + P", font_size=20, color=BLUE)
        X_group = VGroup(X_label, X_matrix).arrange(DOWN, buff=0.2)
        X_group.move_to(RIGHT * 3.5)
        
        # Plus sign
        plus = Text("+", font_size=40).move_to(ORIGIN)
        
        # Animate
        self.play(
            self.E_visual.animate.scale(0.8).move_to(LEFT * 3.5 + UP * 0.3),
            FadeIn(plus),
            FadeIn(P_group),
            run_time=2
        )
        self.wait(0.5)
        self.play(FadeIn(X_group), run_time=1.5)
        
        explanation = Text("Position information encoded as sine/cosine waves", font_size=16, color=GRAY)
        explanation.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(explanation))
        
        self.wait(1)
        self.current_scene = VGroup(self.E_visual, plus, P_group, X_group)
        self.explanation = explanation
    
    def scene_5_qkv_projections(self):
        """Scene 5: Linear projections to Q, K, V"""
        new_title = Text("Scene 5: Q, K, V Projections", font_size=36, color=self.COLOR_Q, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        self.play(FadeOut(self.current_scene), FadeOut(self.explanation))
        
        # Show X matrix
        X_matrix = self.create_matrix_visual(self.X, color=BLUE, scale=0.6)
        X_label = Text("X", font_size=18, color=BLUE)
        X_group = VGroup(X_label, X_matrix).arrange(DOWN, buff=0.15)
        X_group.move_to(LEFT * 5)
        
        # Show three projections
        Q_matrix = self.create_matrix_visual(self.Q, color=self.COLOR_Q, scale=0.7)
        Q_label = Text("Q", font_size=18, color=self.COLOR_Q)
        Q_group = VGroup(Q_label, Q_matrix).arrange(DOWN, buff=0.15)
        Q_group.move_to(UP * 2 + RIGHT * 2)
        
        K_matrix = self.create_matrix_visual(self.K, color=self.COLOR_K, scale=0.7)
        K_label = Text("K", font_size=18, color=self.COLOR_K)
        K_group = VGroup(K_label, K_matrix).arrange(DOWN, buff=0.15)
        K_group.move_to(ORIGIN + RIGHT * 2)
        
        V_matrix = self.create_matrix_visual(self.V, color=self.COLOR_V, scale=0.7)
        V_label = Text("V", font_size=18, color=self.COLOR_V)
        V_group = VGroup(V_label, V_matrix).arrange(DOWN, buff=0.15)
        V_group.move_to(DOWN * 2 + RIGHT * 2)
        
        # Formulas
        q_formula = Text("Q = XW^Q", font_size=16, color=self.COLOR_Q).next_to(Q_group, RIGHT, buff=0.3)
        k_formula = Text("K = XW^K", font_size=16, color=self.COLOR_K).next_to(K_group, RIGHT, buff=0.3)
        v_formula = Text("V = XW^V", font_size=16, color=self.COLOR_V).next_to(V_group, RIGHT, buff=0.3)
        
        # Arrows
        arrow1 = Arrow(X_group.get_right(), Q_group.get_left(), color=self.COLOR_Q, stroke_width=3)
        arrow2 = Arrow(X_group.get_right(), K_group.get_left(), color=self.COLOR_K, stroke_width=3)
        arrow3 = Arrow(X_group.get_right(), V_group.get_left(), color=self.COLOR_V, stroke_width=3)
        
        # Animate
        self.play(FadeIn(X_group))
        self.wait(0.5)
        self.play(
            GrowArrow(arrow1),
            FadeIn(Q_group),
            FadeIn(q_formula),
            run_time=1
        )
        self.play(
            GrowArrow(arrow2),
            FadeIn(K_group),
            FadeIn(k_formula),
            run_time=1
        )
        self.play(
            GrowArrow(arrow3),
            FadeIn(V_group),
            FadeIn(v_formula),
            run_time=1
        )
        
        self.wait(1)
        self.current_scene = VGroup(X_group, arrow1, arrow2, arrow3, Q_group, K_group, V_group, q_formula, k_formula, v_formula)
    
    def scene_6_attention_scores(self):
        """Scene 6: Compute attention scores"""
        new_title = Text("Scene 6: Attention Scores (QK^T)", font_size=36, color=self.COLOR_SCORES, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        self.play(FadeOut(self.current_scene))
        
        # Show computation
        Q_matrix = self.create_matrix_visual(self.Q, color=self.COLOR_Q, scale=0.6)
        Q_label = Text("Q", font_size=16, color=self.COLOR_Q)
        Q_group = VGroup(Q_label, Q_matrix).arrange(DOWN, buff=0.15)
        
        KT_matrix = self.create_matrix_visual(self.K.T, color=self.COLOR_K, scale=0.6)
        KT_label = Text("K^T", font_size=16, color=self.COLOR_K)
        KT_group = VGroup(KT_label, KT_matrix).arrange(DOWN, buff=0.15)
        
        scores_matrix = self.create_matrix_visual(self.scores, color=self.COLOR_SCORES, scale=0.7)
        scores_label = Text("Scores", font_size=16, color=self.COLOR_SCORES)
        scores_group = VGroup(scores_label, scores_matrix).arrange(DOWN, buff=0.15)
        
        equation = VGroup(
            Q_group,
            Text("×", font_size=24),
            KT_group,
            Text("÷√dₖ", font_size=20),
            Text("=", font_size=24),
            scores_group
        ).arrange(RIGHT, buff=0.25)
        equation.move_to(ORIGIN)
        
        self.play(FadeIn(equation), run_time=2)
        
        explanation = Text("Similarity: how much each token attends to others", font_size=18, color=GRAY)
        explanation.next_to(equation, DOWN, buff=0.6)
        self.play(FadeIn(explanation))
        
        self.wait(1)
        self.current_scene = equation
        self.explanation = explanation
    
    def scene_7_softmax(self):
        """Scene 7: Softmax normalization"""
        new_title = Text("Scene 7: Softmax → Attention Weights", font_size=36, color=self.COLOR_WEIGHTS, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        self.play(FadeOut(self.current_scene), FadeOut(self.explanation))
        
        # Show transformation
        scores_matrix = self.create_matrix_visual(self.scores, color=self.COLOR_SCORES, scale=0.7)
        scores_label = Text("Raw Scores", font_size=16, color=self.COLOR_SCORES)
        scores_group = VGroup(scores_label, scores_matrix).arrange(DOWN, buff=0.2)
        
        weights_matrix = self.create_matrix_visual(self.attention_weights, color=self.COLOR_WEIGHTS, scale=0.7)
        weights_label = Text("Attention α", font_size=16, color=self.COLOR_WEIGHTS)
        weights_group = VGroup(weights_label, weights_matrix).arrange(DOWN, buff=0.2)
        
        transformation = VGroup(
            scores_group,
            Text("→ softmax →", font_size=20),
            weights_group
        ).arrange(RIGHT, buff=0.4)
        transformation.move_to(ORIGIN)
        
        self.play(FadeIn(transformation), run_time=2)
        
        explanation = Text("Each row sums to 1.0 (probability distribution)", font_size=18, color=GRAY)
        explanation.next_to(transformation, DOWN, buff=0.6)
        self.play(FadeIn(explanation))
        
        self.wait(1)
        self.current_scene = transformation
        self.explanation = explanation
    
    def scene_8_value_mixing(self):
        """Scene 8: Weighted sum of values"""
        new_title = Text("Scene 8: Value Mixing (Z = αV)", font_size=36, color=self.COLOR_OUTPUT, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        self.play(FadeOut(self.current_scene), FadeOut(self.explanation))
        
        # Show computation
        alpha_matrix = self.create_matrix_visual(self.attention_weights, color=self.COLOR_WEIGHTS, scale=0.6)
        alpha_label = Text("α", font_size=16, color=self.COLOR_WEIGHTS)
        alpha_group = VGroup(alpha_label, alpha_matrix).arrange(DOWN, buff=0.15)
        
        V_matrix = self.create_matrix_visual(self.V, color=self.COLOR_V, scale=0.6)
        V_label = Text("V", font_size=16, color=self.COLOR_V)
        V_group = VGroup(V_label, V_matrix).arrange(DOWN, buff=0.15)
        
        Z_matrix = self.create_matrix_visual(self.Z, color=self.COLOR_OUTPUT, scale=0.7)
        Z_label = Text("Z", font_size=16, color=self.COLOR_OUTPUT)
        Z_group = VGroup(Z_label, Z_matrix).arrange(DOWN, buff=0.15)
        
        equation = VGroup(
            alpha_group,
            Text("×", font_size=24),
            V_group,
            Text("=", font_size=24),
            Z_group
        ).arrange(RIGHT, buff=0.3)
        equation.move_to(ORIGIN)
        
        self.play(FadeIn(equation), run_time=2)
        
        explanation = Text("Each token = weighted combination of all values", font_size=18, color=GRAY)
        explanation.next_to(equation, DOWN, buff=0.6)
        self.play(FadeIn(explanation))
        
        self.wait(1)
        self.current_scene = equation
        self.explanation = explanation
    
    def scene_9_semantic_emergence(self):
        """Scene 9: Final contextual embeddings"""
        new_title = Text("Scene 9: Contextual Meaning Emerges", font_size=36, color=self.COLOR_FINAL, weight=BOLD)
        new_title.to_edge(UP)
        self.play(Transform(self.current_title, new_title))
        
        self.play(FadeOut(self.current_scene), FadeOut(self.explanation))
        
        # Show final output with original tokens
        Z_matrix = self.create_matrix_visual(self.Z, color=self.COLOR_FINAL, scale=1.1)
        Z_matrix.move_to(ORIGIN)
        
        # Add token labels
        token_labels = VGroup()
        for i, token in enumerate(self.tokens):
            label = Text(token, font_size=14, color=GRAY)
            token_labels.add(label)
        token_labels.arrange(DOWN, buff=0.49)
        token_labels.next_to(Z_matrix, LEFT, buff=0.3)
        
        self.play(FadeIn(Z_matrix), FadeIn(token_labels), run_time=2)
        
        # Show original sentence faintly in background
        original = Text(f'"{self.sentence}"', font_size=20, color=GRAY, opacity=0.3)
        original.to_edge(DOWN, buff=1.5)
        self.play(FadeIn(original))
        
        # Final explanation
        explanation = VGroup(
            Text("Context-aware representations", font_size=22, color=self.COLOR_FINAL, weight=BOLD),
            Text("Each vector now contains information from entire sequence", font_size=16, color=GRAY),
            Text("'cat' understands it's doing the 'sitting' action", font_size=16, color=GRAY)
        ).arrange(DOWN, buff=0.25)
        explanation.next_to(Z_matrix, DOWN, buff=0.8)
        self.play(FadeIn(explanation), run_time=2)
        
        self.wait(2)


# Render with:
# manim -pql complete_transformer.py CompleteTransformerPipeline  # Low quality
# manim -pqh complete_transformer.py CompleteTransformerPipeline  # High quality