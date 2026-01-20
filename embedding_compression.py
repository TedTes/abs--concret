from manim import *
import numpy as np

class CompressionVsEmbeddings(Scene):
    def construct(self):
        # Title phase
        title = Text("Compression vs Embeddings", font_size=48, weight=BOLD)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))
        
        # Create dividing line
        divider = Line(UP * 3.5, DOWN * 3.5, color=GRAY).shift(RIGHT * 0.1)
        
        # Labels - positioned higher to avoid overlap
        comp_label = Text("Compression", font_size=32, color="#FF6B6B").move_to(UP * 3.5 + LEFT * 3.5)
        emb_label = Text("Embeddings", font_size=32, color="#4ECDC4").move_to(UP * 3.5 + RIGHT * 3.5)
        
        self.play(
            Create(divider),
            Write(comp_label),
            Write(emb_label)
        )
        self.wait(0.5)
        
        # SCENE 1: Input data (same for both)
        words = ["king", "queen", "man", "woman", "cat", "dog"]
        
        # Left side: Compression - smaller font to reduce overlap
        comp_input = VGroup(*[
            Text(word, font_size=24, color=WHITE)
            for word in words
        ]).arrange(DOWN, buff=0.25).move_to(LEFT * 3.5 + UP * 1.8)
        
        # Right side: Embeddings - same input
        emb_input = comp_input.copy().move_to(RIGHT * 3.5 + UP * 1.8)
        
        self.play(
            FadeIn(comp_input),
            FadeIn(emb_input)
        )
        self.wait(1)
        
        # SCENE 2: Show the question each answers
        comp_question = Text("How to describe\nexactly with\nfewer bits?", 
                           font_size=18, color="#FF6B6B", 
                           line_spacing=0.7).move_to(LEFT * 3.5 + UP * 0.2)
        
        emb_question = Text("What relationships\nmatter enough\nto preserve?", 
                          font_size=18, color="#4ECDC4",
                          line_spacing=0.7).move_to(RIGHT * 3.5 + UP * 0.2)
        
        self.play(
            FadeIn(comp_question, shift=UP * 0.2),
            FadeIn(emb_question, shift=UP * 0.2)
        )
        self.wait(2)
        self.play(FadeOut(comp_question), FadeOut(emb_question))
        
        # SCENE 3: Compression - encode to smaller representation
        comp_compressed = Rectangle(
            width=1.8, height=1.2, 
            fill_color="#FF6B6B", fill_opacity=0.3,
            stroke_color="#FF6B6B"
        ).move_to(LEFT * 3.5 + DOWN * 0.3)
        
        comp_bits = VGroup(*[
            Text("01", font_size=14, color="#FF6B6B")
            for _ in range(9)
        ]).arrange_in_grid(rows=3, cols=3, buff=0.12).move_to(comp_compressed)
        
        compress_arrow = Arrow(
            comp_input.get_bottom() + DOWN * 0.05,
            comp_compressed.get_top() + UP * 0.05,
            color="#FF6B6B", buff=0.05, stroke_width=3
        )
        
        compress_label = Text("encode", font_size=16, color="#FF6B6B").next_to(compress_arrow, LEFT, buff=0.15)
        
        self.play(
            GrowArrow(compress_arrow),
            FadeIn(compress_label),
            Create(comp_compressed),
            FadeIn(comp_bits)
        )
        self.wait(1)
        
        # SCENE 4: Embeddings - transform to semantic space
        # Create 2D semantic space visualization - repositioned and resized
        axes = Axes(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=2.5,
            y_length=2.5,
            axis_config={"include_tip": False, "stroke_opacity": 0.2, "stroke_width": 1},
        ).move_to(RIGHT * 3.5 + DOWN * 0.5)
        
        # Semantic positions (relationships preserved) - adjusted for smaller axes
        semantic_positions = {
            "king": np.array([0.5, 0.4, 0]),
            "queen": np.array([0.4, 0.5, 0]),
            "man": np.array([0.35, -0.25, 0]),
            "woman": np.array([0.25, -0.15, 0]),
            "cat": np.array([-0.4, 0.25, 0]),
            "dog": np.array([-0.3, 0.35, 0])
        }
        
        embed_dots = VGroup()
        embed_labels = VGroup()
        
        for word in words:
            pos = axes.c2p(*semantic_positions[word])
            dot = Dot(pos, color="#4ECDC4", radius=0.06)
            # Position labels to avoid overlap - alternate sides
            if word in ["king", "cat"]:
                label = Text(word, font_size=12, color="#4ECDC4").next_to(dot, LEFT, buff=0.08)
            elif word in ["queen", "dog"]:
                label = Text(word, font_size=12, color="#4ECDC4").next_to(dot, RIGHT, buff=0.08)
            elif word == "man":
                label = Text(word, font_size=12, color="#4ECDC4").next_to(dot, DOWN, buff=0.08)
            else:  # woman
                label = Text(word, font_size=12, color="#4ECDC4").next_to(dot, UP, buff=0.08)
            embed_dots.add(dot)
            embed_labels.add(label)
        
        embed_arrow = Arrow(
            emb_input.get_bottom() + DOWN * 0.05,
            axes.get_top() + UP * 0.05,
            color="#4ECDC4", buff=0.05, stroke_width=3
        )
        
        embed_label_text = Text("transform", font_size=16, color="#4ECDC4").next_to(embed_arrow, RIGHT, buff=0.15)
        
        self.play(
            GrowArrow(embed_arrow),
            FadeIn(embed_label_text),
            Create(axes)
        )
        self.play(
            FadeIn(embed_dots),
            FadeIn(embed_labels)
        )
        self.wait(1)
        
        # SCENE 5: Show relationships in embedding space
        # Highlight semantic clusters - smaller circles to avoid overlap
        royalty_circle = Circle(radius=0.35, color="#95E1D3", stroke_width=2).move_to(
            axes.c2p(0.45, 0.45, 0)
        )
        gender_circle = Circle(radius=0.28, color="#95E1D3", stroke_width=2).move_to(
            axes.c2p(0.3, -0.2, 0)
        )
        animals_circle = Circle(radius=0.28, color="#95E1D3", stroke_width=2).move_to(
            axes.c2p(-0.35, 0.3, 0)
        )
        
        relationship_text = Text("relationships\npreserved", font_size=14, color="#95E1D3", 
                                line_spacing=0.7).next_to(axes, DOWN, buff=0.25)
        
        self.play(
            Create(royalty_circle),
            Create(gender_circle),
            Create(animals_circle),
            FadeIn(relationship_text)
        )
        self.wait(1.5)
        self.play(FadeOut(relationship_text))
        
        # SCENE 6: Compression - attempt to decode
        decompress_arrow = Arrow(
            comp_compressed.get_bottom() + DOWN * 0.05,
            comp_compressed.get_bottom() + DOWN * 1.2,
            color="#FF6B6B", buff=0.05, stroke_width=3
        )
        
        decode_label = Text("decode", font_size=16, color="#FF6B6B").next_to(decompress_arrow, LEFT, buff=0.15)
        
        # Reconstructed output (with potential errors) - smaller font
        comp_output = VGroup(*[
            Text(word if i < 4 else word + "?", font_size=20, 
                 color=WHITE if i < 4 else "#FFB6B6")
            for i, word in enumerate(words)
        ]).arrange(DOWN, buff=0.2).move_to(LEFT * 3.5 + DOWN * 2.2)
        
        self.play(
            GrowArrow(decompress_arrow),
            FadeIn(decode_label)
        )
        self.play(FadeIn(comp_output))
        self.wait(1)
        
        # SCENE 7: Key difference - Truth vs Usefulness
        truth_text = Text("keeps TRUTH\n(exactness)", font_size=20, color="#FF6B6B",
                         line_spacing=0.6).move_to(LEFT * 3.5 + DOWN * 3.3)
        
        useful_text = Text("keeps USEFULNESS\n(meaning)", font_size=20, color="#4ECDC4",
                          line_spacing=0.6).move_to(RIGHT * 3.5 + DOWN * 2.3)
        
        self.play(
            FadeIn(truth_text, shift=UP * 0.2),
            FadeIn(useful_text, shift=UP * 0.2)
        )
        self.wait(2)
        
        # SCENE 8: The Gap - Where Intelligence Lives
        self.play(
            FadeOut(comp_input), FadeOut(emb_input),
            FadeOut(compress_arrow), FadeOut(compress_label),
            FadeOut(comp_compressed), FadeOut(comp_bits),
            FadeOut(decompress_arrow), FadeOut(decode_label),
            FadeOut(comp_output),
            FadeOut(embed_arrow), FadeOut(embed_label_text),
            FadeOut(axes), FadeOut(embed_dots), FadeOut(embed_labels),
            FadeOut(royalty_circle), FadeOut(gender_circle), FadeOut(animals_circle),
            FadeOut(truth_text), FadeOut(useful_text),
            FadeOut(comp_label), FadeOut(emb_label),
            FadeOut(divider)
        )
        self.wait(0.5)
        
        # Final message
        gap_line1 = Text("That gap", font_size=42, color="#F38181", weight=BOLD)
        gap_line2 = Text("is where intelligence lives", font_size=42, color="#F38181", weight=BOLD)
        gap_group = VGroup(gap_line1, gap_line2).arrange(DOWN, buff=0.5)
        
        # Create visual representation of "the gap" - better spacing
        left_box = Rectangle(width=2.2, height=3.5, fill_opacity=0.2, fill_color="#FF6B6B", 
                            stroke_color="#FF6B6B").shift(LEFT * 2.8)
        right_box = Rectangle(width=2.2, height=3.5, fill_opacity=0.2, fill_color="#4ECDC4",
                             stroke_color="#4ECDC4").shift(RIGHT * 2.8)
        
        gap_region = Rectangle(width=1.8, height=3.5, fill_opacity=0.4, fill_color="#F38181",
                              stroke_color="#F38181", stroke_width=3)
        
        left_label = Text("exact\nbits", font_size=22, color="#FF6B6B", 
                         line_spacing=0.7).move_to(left_box)
        right_label = Text("semantic\nspace", font_size=22, color="#4ECDC4",
                          line_spacing=0.7).move_to(right_box)
        
        self.play(
            Create(left_box),
            Create(right_box),
            FadeIn(left_label),
            FadeIn(right_label)
        )
        self.wait(0.5)
        
        self.play(
            Create(gap_region),
            Write(gap_group.move_to(UP * 2.2))
        )
        self.wait(3)
        
        # Final fade
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )
        self.wait(0.5)


class CompressionVsEmbeddingsMinimal(Scene):
    """Cleaner minimal version with no overlaps"""
    def construct(self):
        # SETUP: Two parallel paths
        divider = DashedLine(UP * 4, DOWN * 4, color=GRAY_C).shift(RIGHT * 0.1)
        
        # Starting point - same input
        input_text = Text("DATA", font_size=40, color=WHITE)
        
        self.play(Write(input_text))
        self.wait(0.5)
        
        # Split into two copies
        left_input = input_text.copy().move_to(LEFT * 3.5 + UP * 2.5)
        right_input = input_text.copy().move_to(RIGHT * 3.5 + UP * 2.5)
        
        self.play(
            Transform(input_text, VGroup(left_input, right_input)),
            Create(divider)
        )
        self.wait(0.5)
        
        # LEFT PATH: Compression - Show as binary reduction
        binary_rect = Rectangle(width=1.8, height=0.9, fill_color="#FF6B6B", 
                               fill_opacity=0.3, stroke_color="#FF6B6B")
        binary_rect.move_to(LEFT * 3.5 + UP * 0.5)
        binary_pattern = Text("010110", font_size=22, color="#FF6B6B").move_to(binary_rect)
        
        arrow_down_left = Arrow(left_input.get_bottom() + DOWN * 0.1, 
                               binary_rect.get_top() + UP * 0.1, 
                               color="#FF6B6B", buff=0.1, stroke_width=3)
        
        self.play(
            GrowArrow(arrow_down_left),
            Create(binary_rect),
            Write(binary_pattern)
        )
        self.wait(0.5)
        
        # Reconstruct (with error indicator)
        reconstructed = Text("DATA?", font_size=34, color="#FFB6B6").move_to(LEFT * 3.5 + DOWN * 1.5)
        arrow_down_left2 = Arrow(binary_rect.get_bottom() + DOWN * 0.1, 
                                reconstructed.get_top() + UP * 0.1,
                                color="#FF6B6B", buff=0.1, stroke_width=3)
        
        exact_label = Text("exact reconstruction", font_size=16, color="#FF6B6B").next_to(reconstructed, DOWN, buff=0.3)
        
        self.play(
            GrowArrow(arrow_down_left2),
            Write(reconstructed)
        )
        self.play(FadeIn(exact_label))
        self.wait(1)
        
        # RIGHT PATH: Embeddings - Show as geometric space
        dots_positions = [
            RIGHT * 3.1 + UP * 0.6,
            RIGHT * 3.9 + UP * 0.8,
            RIGHT * 3.3 + UP * 0.0,
            RIGHT * 3.8 + UP * 0.1
        ]
        
        dots = VGroup(*[Dot(pos, color="#4ECDC4", radius=0.09) for pos in dots_positions])
        
        # Connection lines showing relationships
        lines = VGroup(
            Line(dots[0].get_center(), dots[1].get_center(), color="#95E1D3", stroke_width=2),
            Line(dots[2].get_center(), dots[3].get_center(), color="#95E1D3", stroke_width=2)
        )
        
        arrow_down_right = Arrow(right_input.get_bottom() + DOWN * 0.1, 
                                dots.get_top() + UP * 0.4,
                                color="#4ECDC4", buff=0.1, stroke_width=3)
        
        self.play(GrowArrow(arrow_down_right))
        self.play(
            FadeIn(dots),
            Create(lines)
        )
        self.wait(0.5)
        
        # Show preserved relationships
        meaning_label = Text("preserved relationships", font_size=16, color="#4ECDC4").move_to(RIGHT * 3.5 + DOWN * 1.5)
        
        self.play(FadeIn(meaning_label))
        self.wait(1)
        
        # FINALE: Highlight the difference
        truth = Text("TRUTH", font_size=30, color="#FF6B6B", weight=BOLD).move_to(LEFT * 3.5 + DOWN * 2.8)
        usefulness = Text("USEFULNESS", font_size=30, color="#4ECDC4", weight=BOLD).move_to(RIGHT * 3.5 + DOWN * 2.8)
        
        self.play(
            FadeOut(exact_label),
            FadeOut(meaning_label),
            Write(truth),
            Write(usefulness)
        )
        self.wait(2)
        
        # The gap
        self.play(*[FadeOut(mob) for mob in [left_input, right_input, arrow_down_left, 
                                              binary_rect, binary_pattern, arrow_down_left2,
                                              reconstructed, arrow_down_right, dots, lines]])
        
        gap_text = Text("The gap is where\nintelligence lives", 
                       font_size=38, color="#F38181", weight=BOLD,
                       line_spacing=0.8)
        
        # Animate gap growing between truth and usefulness
        self.play(
            truth.animate.shift(LEFT * 1.8),
            usefulness.animate.shift(RIGHT * 1.8)
        )
        
        gap_region = Rectangle(width=3.5, height=2, fill_color="#F38181", 
                              fill_opacity=0.3, stroke_color="#F38181")
        
        self.play(Create(gap_region))
        self.play(Write(gap_text))
        self.wait(3)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])