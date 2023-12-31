CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230730000000_e20230730235959_p20230731021641_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-31T02:16:41.995Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-30T00:00:00.000Z   time_coverage_end         2023-07-30T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�A�   �          A=q@u��z�@�ffB6(�C��
@u�u@�=qBEQ�C��f                                    Bx�A��  �          A@z=q���@�33B1�HC���@z=q�|(�@�\)BA(�C�^�                                    Bx�A�L  �          A ��@p  ���@�Q�A��\C�j=@p  ��@���B	��C�T{                                    Bx�A��  �          @��@Vff��=q@XQ�AծC���@Vff���@x��A��C��3                                    Bx�A��  T          @�z�@i����@vffB�C��
@i���s�
@�
=B��C��=                                    Bx�A�>  �          @�=q?\@G
=@�G�Bp��B��?\@i��@�\)B]��B��R                                    Bx�A��  �          @�{?�
=@%�@��B��HBs�H?�
=@I��@���Bp�HB��                                    Bx�A�  �          @�?}p�@�(�@��HBA{B��?}p�@�(�@��B,Q�B�(�                                    Bx�A�0  �          @��?W
=@��H@��B;33B�W
?W
=@�=q@�  B&G�B��                                    Bx�B�  �          @�G�?n{@��
@�p�BO  B���?n{@�(�@��B:(�B�k�                                    Bx�B|  �          @�?�@b�\@��
Bj�RB�L�?�@��H@���BV\)B�                                      Bx�B%"  �          @�(�?�{@XQ�@θRBt�B��R?�{@}p�@ÅB`33B���                                    Bx�B3�  T          @�=q?�p�@<��@ҏ\B�(�B�Ǯ?�p�@c33@���Bl=qB���                                    Bx�BBn  �          @�=q?�p�@9��@�G�B~=qB|{?�p�@`  @ǮBj�RB�B�                                    Bx�BQ  
�          @�(�?p��@`��@�z�Bm\)B��?p��@�=q@���BXp�B�\                                    Bx�B_�  T          @��?�  @e�@�  Bi
=B��3?�  @�(�@�z�BT�B��3                                    Bx�Bn`  "          @�(�?�
=@7�@��
B|Q�B�8R?�
=@\(�@�=qBh
=B�                                      Bx�B}  �          @��
?s33@O\)@�\)BrQ�B�8R?s33@r�\@���B]Q�B�Ǯ                                    Bx�B��  
�          @���?Q�@�@�G�B�#�B�p�?Q�@:�H@�G�B}p�B�Ǯ                                    Bx�B�R  T          @��
?�  @333@�\)Bt33Bh�?�  @W
=@�Ba33By��                                    Bx�B��  "          @�@�\=u@�33B��R?�@�\?333@�=qB��RA���                                    Bx�B��  �          @��@z�?�  @�Q�B�
=A��
@z�?˅@�z�B��3B	�R                                    Bx�B�D  T          @�p�@�    @ָRB��C��R@�?#�
@�B���As
=                                    Bx�B��  
�          @��
?��H?�  @�Q�B�(�AظR?��H?�@�(�B���B"
=                                    Bx�B�  �          @�{?���?���@�
=B��3BB�H?���@\)@أ�B���Be{                                    Bx�B�6  �          @�  @2�\�B�\@��B�W
C��@2�\?   @�z�B��RA$(�                                    Bx�C �  �          @��@%��\)@�B���C�3@%�#�
@���B���C��                                    Bx�C�  "          A ��@(���G�@�=qB�#�C�Ff@(�ÿ�ff@�\)B�{C���                                    Bx�C(  �          A{@(���p�@�  B��\C�|)@(���p�@���B��3C���                                    Bx�C,�  T          AG�@33���@�B���C�%@33���H@�B���C�Ф                                    Bx�C;t  �          @�\)?�\)��R@�33B�.C�� ?�\)�޸R@��B���C�}q                                    Bx�CJ  T          A ��@ff���@��B��C���@ff��33@�B�W
C��                                    Bx�CX�  �          @��
@33�E�@�p�Bwz�C�Ф@33�Q�@�{B��
C�^�                                    Bx�Cgf  �          @�@#�
�5�@�ffBtC��@#�
�Q�@�{B�L�C�%                                    Bx�Cv  �          @��@ ���HQ�@��Bo�\C�Y�@ ����H@�{B�B�C��)                                    Bx�C��  "          A ��@.�R�^{@�Bb33C��@.�R�1�@�  Bs�C�5�                                    Bx�C�X  �          A@��#33@�B�z�C���@���ff@�\B��RC��{                                    Bx�C��  �          A
=?�(���@�G�B���C��H?�(����
@�ffB�p�C�o\                                    Bx�C��  �          A\)?�  ��\)@�{B�{C��?�  ���AG�B��{C��q                                    Bx�C�J  �          A\)@I�����H@�(�BL{C��@I���Z�H@�Q�B]�C�P�                                    Bx�C��  �          AG�@A��\)@ʏ\BN��C��R@A��Tz�@�ffB`��C�(�                                    Bx�Cܖ  �          A�@<���u@�  BUG�C��@<���J=q@ۅBg33C��=                                    Bx�C�<  �          A{@>�R�O\)@ڏ\Bd�RC�K�@>�R�!�@��
Bu{C��)                                    Bx�C��  "          A�@7��C�
@��Bm33C���@7��z�@��HB}G�C��f                                    Bx�D�  "          A\)@6ff�S33@޸RBgC�n@6ff�#�
@�Q�Bx�C��                                    Bx�D.  
�          A�\@0  �XQ�@�z�Bf�
C��=@0  �)��@�RBxffC��                                    Bx�D%�  �          A�@:=q��Q�@�G�BS�C�  @:=q�S33@�p�Bf�C��                                    Bx�D4z  �          A��@:�H���\@�33BN��C�Ǯ@:�H�Y��@�  Ba��C�T{                                    Bx�DC   "          A(�@N{����@���BLG�C�:�@N{�W
=@�G�B^ffC��                                    Bx�DQ�  �          A   @p���p�@�ffB33C�� @p���p�@�=qBp�C��)                                    Bx�D`l  
�          A   @(Q����
@�B��C��H@(Q����
@���B��C���                                    Bx�Do  "          A  @!�����@�(�A�
=C�c�@!���@��B{C�q                                    Bx�D}�  �          A�R@p���Q�@UA��C���@p��Ӆ@��A�p�C�8R                                    Bx�D�^  �          @�G�?�33���
@ϮB`��C�g�?�33�Z=q@�z�Bw�C�1�                                    Bx�D�  T          @�p�?�=q���@�G�B,�C��?�=q��33@��BCffC��                                     Bx�D��  T          @�?������@�(�B0�C���?����
=@���BGffC��                                    Bx�D�P            @�(�?s33��33@�B��C���?s33���@�  B6=qC�U�                                    Bx�D��  �          @�R?Q���33@��
B8z�C��3?Q����@�z�BP�HC�.                                    Bx�D՜  �          @�p�?��
��Q�@��RBffC�>�?��
��  @��B,��C��
                                    Bx�D�B  �          @��@:�H����@R�\A�p�C�ff@:�H��z�@y��B��C�9�                                    Bx�D��  "          @�  @@����Q�@dz�A���C�W
@@�����H@���B�C�Q�                                    Bx�E�  �          @�@>{���R@�G�B��C��
@>{��\)@��\B�C�
=                                    Bx�E4  �          @�\)?޸R��=q@uB�C�N?޸R���@�ffB33C��R                                    Bx�E�  �          @�\@ �����
@z=qB ��C�n@ ������@���B��C�,�                                    Bx�E-�  �          @�R@&ff��Q�@�
=B>ffC���@&ff�hQ�@��BS��C�˅                                    Bx�E<&  �          @�R@!����@���B?p�C�7
@!��j�H@��RBT�C�K�                                    Bx�EJ�  "          @�?ٙ���Q�@P  A��C��?ٙ����@y��B�C�j=                                    Bx�EYr  �          @�G�?�z���=q@k�A��\C���?�z����
@��B(�C�]q                                    Bx�Eh  T          @�\>���  @��\B  C�K�>���Q�@�{B&��C���                                    Bx�Ev�  �          @��H?z���z�@�ffB{C��
?z���(�@���B+��C�AH                                    Bx�E�d  T          @�G�?�p���  @k�A�(�C�1�?�p�����@���B�C�Ǯ                                    Bx�E�
  
�          @޸R@ ���Y��@�p�B\�\C�E@ ���/\)@�Q�BrQ�C�                                      Bx�E��  C          @�p�@��{@���B6ffC���@�g
=@�33BL��C�xR                                    Bx�E�V  3          @�G�@)������@J=qA�=qC�J=@)�����@o\)B\)C�1�                                    Bx�E��  T          @�=q?�z���(��8Q�\C�Q�?�z��˅?\)@�G�C�XR                                    Bx�E΢  �          @��?�33�ҏ\��{�X��C��
?�33��
=�W
=��G�C��)                                    Bx�E�H  �          @��?���  �h����\)C�1�?���녾�����C�!H                                    Bx�E��  �          @�G�?�
=��  @�A�C�?�
=��
=@*=qA��HC���                                    Bx�E��  �          @�?�����@��B�W
C���?��
=@�z�B��3C��                                    Bx�F	:  �          @�\@33�333@�z�B�8RC���@33>�@�p�B�
=@n{                                    Bx�F�  �          @�{?ٙ���(�@��B��C��{?ٙ���R@�G�B�
=C���                                    Bx�F&�  "          @���@��?O\)@��
B��A�G�@��?�@�
=B��B��                                    Bx�F5,  "          @�Q�@(Q�?��@�RB���A��R@(Q�?�Q�@���B�8RB��                                    Bx�FC�  
�          A   @>�R�0��@�33B��HC�� @>�R>B�\@�(�B�  @h��                                    Bx�FRx  �          A�@0�׿�{@���B��C��@0�׾L��@�B�\C��H                                    Bx�Fa  T          @��\@A녿Ǯ@�{B�C�Z�@A녿8Q�@�\B��{C�L�                                    Bx�Fo�  T          @�{@'
=�y��@�
=BEQ�C��@'
=�N{@�z�B[�C���                                    Bx�F~j  �          @�
=?�Q�����?��RAN�\C�@ ?�Q���@  A���C�p�                                    Bx�F�  �          @��?�
=��{@
=A���C���?�
=���@C�
Aޣ�C�*=                                    Bx�F��  �          @�
=?��\��p�@q�B��C���?��\��p�@��B#�HC�9�                                    Bx�F�\  "          @�G�?�z��g
=@�=qBRffC���?�z��>{@�
=Bl�C���                                    Bx�F�  T          @���?޸R�-p�@�G�Bs��C�Z�?޸R��p�@\B�\C���                                    Bx�FǨ  �          @��?�
=�6ff@��\Bu{C�T{?�
=��@�(�B��RC�
=                                    Bx�F�N  �          @�
=?��*�H@�33Bo�
C�AH?����H@�z�B���C��\                                    Bx�F��  
�          @׮@�R�z�@���Bu�RC��@�R����@���B�(�C�\)                                    Bx�F�  
�          @��@   �33@��HBy��C���@   �Ǯ@\B��qC�{                                    Bx�G@  
�          @��?�ff���@�  B��C��H?�ff��33@�{B�
=C���                                    Bx�G�  �          @�@��333@��
B�.C��{@�>�@��B���@E                                    Bx�G�  T          @�@��>�Q�@���B��RA�H@��?�p�@�B�ǮA��
                                    Bx�G.2  �          @�(�?��?���@ᙚB��A�  ?��?���@ۅB��B8�                                    Bx�G<�  T          @���@33>�Q�@�Q�B�z�A Q�@33?�
=@��B�p�A�                                    Bx�GK~  "          @�\)@,�Ϳc�
@��HB���C��)@,�ͽ��
@���B�(�C�!H                                    Bx�GZ$  T          @�=q?���?��
@أ�B��B  ?���?��@ҏ\B�{BH�\                                    Bx�Gh�  �          @�\)?�?�33@��B�ffBH  ?�@\)@�B��Byff                                    Bx�Gwp  �          @�?��H��ff@&ffA�G�C��H?��H���\@Q�A�z�C��                                    Bx�G�  "          @���@(Q��=q@�\)BYz�C�˅@(Q�޸R@�  BkC�=q                                    Bx�G��  �          @�z�@�p��W
=@��\B/�C��@�p��u@���B3Q�C��=                                    Bx�G�b  
�          @�\)@Fff�	��@���B^��C���@Fff��z�@�  Bm��C���                                    Bx�G�  �          @�33?˅�3�
@�  BvC���?˅�   @��B��
C�>�                                    Bx�G��  �          @�z�?�{�|(�@���BJG�C���?�{�Mp�@���Bd�C��                                    Bx�G�T  �          @���@%��O\)@�p�BR��C�@ @%��   @���Bi
=C��\                                    Bx�G��  �          @أ�?�\)�6ff@�Bp{C���?�\)�z�@�  B���C��                                    Bx�G�  "          @ҏ\?5�Tz�@��\Bj�RC�?5�#33@�
=B��C�˅                                    Bx�G�F  D          @�?����Q�@��
BN�RC��H?���R�\@��Bl�HC���                                    Bx�H	�  �          @�
=?˅�3�
@�33Bt=qC�Ǯ?˅� ��@�p�B��)C�:�                                    Bx�H�  �          @�G�>�(���@�G�BN�C��3>�(��\(�@���Bl��C���                                    Bx�H'8  �          @�
=?�  ���\@��B>��C�&f?�  �vff@�\)B]  C�Ff                                    Bx�H5�  �          @�G�?�G����R@��HBJffC�XR?�G��\��@��Bg��C��                                    Bx�HD�  �          @�{@
=�p  @��BH�RC�3@
=�@  @���Bb33C�R                                    Bx�HS*  �          @�33@���J�H@��B_��C��@���Q�@�G�BxG�C�f                                    Bx�Ha�  �          @�{@����p�@�33B,z�C���@���p  @��BH=qC��H                                    Bx�Hpv  �          @�
=?(���ʏ\@7�A�ffC��{?(����(�@mp�B z�C�.                                    Bx�H  
�          @�{?�
=��=q@>�RA��C�?�
=���@k�BQ�C���                                    Bx�H��  �          @\?޸R����?�  A>ffC���?޸R��p�@ ��A�C��H                                    Bx�H�h  �          @Å?��R���?���AQG�C�n?��R���
@
=A�{C���                                    Bx�H�  
�          @ȣ�?����@�A�C�g�?�����H@333A�  C�˅                                    Bx�H��  �          @��?�{���
?��RA��\C���?�{��G�@2�\A�Q�C�q                                    Bx�H�Z  
�          @�=q?������@	��A��HC�b�?����{@<(�Aݙ�C��=                                    Bx�H�   
�          @���?�G���
=@!�A���C�7
?�G����@S�
A��
C��                                    Bx�H�  T          @љ�?aG��c33@�  B\p�C��)?aG��1�@�ffB{�C��{                                    Bx�H�L  �          @У�?E��   @�  B�ǮC���?E�����@�\)B�L�C�0�                                    Bx�I�  
�          @��@QG����������C��
@QG����H���
�:�HC���                                    Bx�I�  �          @˅@P����=q?}p�A�C���@P����(�?޸RA
=C�5�                                    Bx�I >  T          @�G�@]p���(�?�{A"ffC�  @]p���p�?�=qA��C���                                    Bx�I.�  �          @ȣ�@5����?��AA�C��q@5���@z�A�33C�k�                                    Bx�I=�  �          @���@�����H�8Q���C�޸@����z�<��
>uC�                                    Bx�IL0  �          @ҏ\@�����@&ffA���C��f@����H@W�A���C���                                    Bx�IZ�  �          @�R@��-p�@ə�Btp�C�n@���G�@��
B��C���                                    Bx�Ii|  �          @�(�@{�Q�@ۅB���C��f@{>B�\@��B�=q@��                                    Bx�Ix"  �          @�\)@�
?��
@��
B�aHA�z�@�
@Q�@�(�B�=qB8(�                                    Bx�I��  
�          @�(�@\)?��@�(�B���A��@\)@�@���B��
B!z�                                    Bx�I�n  �          @��@7
=?���@�\)B�#�A���@7
=@   @�Bu{B$��                                    Bx�I�  "          @���@5?��
@�B}\)A��
@5@0  @��HBh=qB0                                      Bx�I��  
�          @���?����y��@�  BV�C�'�?����@  @�Q�Bu��C��                                    Bx�I�`  �          @��?\��
=@Q�AظRC��)?\���@�p�B�C���                                    Bx�I�  T          @�\?fff��\)@L(�A��C�)?fff��@��HB�C��                                     Bx�Iެ  �          @�=q?   ��z�?O\)@�Q�C�n?   ��ff?�
=A�
=C���                                    Bx�I�R  
�          @���\�0�����v33C��H��\�`����ff�S�C��)                                    Bx�I��  v          @��ͿE��mp������N�C�(��E���z����,��C�                                    Bx�J
�  d          @��H��
=�QG����\�d  C�XR��
=��Q������A33C��                                    Bx�JD  �          @��
��z��5�����x�C�쾔z��hQ������U��C��{                                    Bx�J'�  T          @���.{����p�p�C�p��.{�1G�����y�C�8R                                    Bx�J6�  �          @��ÿ����Q���C�q���L�����\�j�\C��                                    Bx�JE6  �          @ȣ׾�\)��ff���HaHC�����\)�{�����C�u�                                    Bx�JS�  �          @�33=��
�c�
����£�HC��q=��
������=q\C�AH                                    Bx�Jb�  �          @��?�(�����3C�  ?�(��>�R��G��o(�C�
                                    Bx�Jq(  �          @���>�  ��ff���G�C���>�  �G�����z�C�/\                                    Bx�J�  T          @��H���H��������z�C6����H��\)��G�CQ��                                    Bx�J�t  �          @�33��{�\)��33  CG
=��{�˅��33C`��                                    Bx�J�  
�          @��!G���33��33��C{�=�!G��5��\)�|(�C��{                                    Bx�J��  
�          @�\)�\�"�\��z��C���\�\(�����e��C��
                                    Bx�J�f  �          @љ������<����=q�{�RC��;����u�����W=qC���                                    Bx�J�  
�          @�p���z�������8RCr�q��z��Mp���
=�l��Cz�                                    Bx�Jײ  �          @�(��
=�(���(��C�=q�
=�[�����n  C�&f                                    Bx�J�X  "          @�ff>L����33�C�
��G�C�>L����������33C��3                                    Bx�J��  "          @Ϯ>�33����^{�G�C��)>�33�����#33��Q�C���                                    Bx�K�  
�          @�ff<#�
��
=�|(��%\)C��<#�
�����J=q� 
=C�\                                    Bx�KJ  �          @��R�\)���\�S�
�	{C��{�\)�����R�ǮC���                                    Bx�K �  �          @��������I����  C��R�������G���ffC�5�                                    Bx�K/�  "          @������  �Vff��HC�<)����\)�\)�\C�z�                                    Bx�K><  
�          @�=q�B�\��z��8����  C��ÿB�\������\��
=C��                                    Bx�KL�  
�          @�=q��
=����Z=q�
33C����
=�����$z���Q�C��q                                    Bx�K[�  
Z          @��Ϳ�(���
=�u��33C�R��(���G��<����  C��f                                    Bx�Kj.  �          @��
�:�H��
=�����F�C���:�H�����Q�� �C�ٚ                                    Bx�Kx�  "          @�������b�\��Q��_�C|O\����������H�:�C��                                    Bx�K�z  "          @�z����,(���Q�8RC������j=q��\)�`=qC�XR                                    Bx�K�   �          @�ff�#�
�r�\����[��C�5ÿ#�
��������5��C�R                                    Bx�K��  �          @�
=�J=q������Q��@�
C��\�J=q��{��{�z�C��\                                    Bx�K�l  �          @ڏ\�5�}p���\)�V�
C��3�5���H��\)�0Q�C�ٚ                                    Bx�K�  "          @�Q��  ���H��=q�9�Cz�H��  ���H��Q��(�C}��                                    Bx�Kи  "          @�G��=q���������(ffCqE�=q����mp��Q�Ct��                                    Bx�K�^  �          @ٙ���ff�|(���{�T�\C���ff������.(�C��{                                    Bx�K�  �          @ҏ\���?\)���H�j(�Cp�Ϳ��y������F�Cvٚ                                    Bx�K��  �          @أ׿���~�R�����P
=C~�������H�����)�\C�]q                                    Bx�LP  �          @�=q�����  ��(��1�C��3������
������C�ٚ                                    Bx�L�  �          @˅=�\)��z����
�&p�C�u�=�\)��G��P  ���RC�g�                                    Bx�L(�  �          @�ff�\)���
��(��4�C���\)��=q�c33���C�9�                                    Bx�L7B  "          @����(��`  �����U�HC�\�(���G�����.  C��{                                    Bx�LE�  �          @���G���p��p  ��C��G���Q��:=q���C���                                    Bx�LT�  T          @�\)�����X����{�b�C�s3������  ��Q��:p�C��
                                    Bx�Lc4  T          @��H?��R�p����(��K��C�#�?��R��=q��(��$�C��q                                    Bx�Lq�  �          @�{?�33�������H�5�C�f?�33�����p  �z�C�                                    Bx�L��  �          @�G�?�z����\�vff��\C��H?�z���{�8Q��љ�C���                                    Bx�L�&  �          @�  �L�����\�u��  C��q�L����{�7
=�ՅC���                                    Bx�L��  �          @��
�L�������
=�+C���L�������b�\�\)C���                                    Bx�L�r  �          @ҏ\�:�H��\)�����E��C�)�:�H��=q��p��C��                                    Bx�L�  
�          @׮�aG���p������@�\C�e�aG���  ��z���\C�E                                    Bx�Lɾ  �          @�녿�z���
=��  �=G�Cn��z�����������C��3                                    Bx�L�d  
�          @�  ���\��p������ffC�q���\����8Q���  C��{                                    Bx�L�
  
�          @陚�8Q����
�����C�=q�8Q���ff��\)��HC�Ff                                    Bx�L��  
Z          @�z�?����=q�����ffC�Ff?���ƸR�;��ǅC��q                                    Bx�MV  
�          @�׿�����p���33���C�#׿������
�O\)��G�C��                                     Bx�M�  
(          @��H���
��{�j=q��(�C�J=���
��Q��p���{C��                                    Bx�M!�  
Z          @陚��=q���\����(�C�G���=q����W�����C�
=                                    Bx�M0H  �          @���\�����p��;(�Cu33��\��G���Q����Cx�3                                    Bx�M>�  "          @ҏ\�p��S33���\�K�
Ci@ �p���{���
�'�Co�=                                    Bx�MM�  T          @ָR�5��P  �����HG�Ce
=�5������ff�%��Ck�{                                    Bx�M\:  T          @�G��@���$z���{�W�C\n�@���b�\����9�Ce��                                    Bx�Mj�  
�          @���I���\)�����p�C>  �I���˅��33�b33CN��                                    Bx�My�  T          @˅�B�\��R��G��s
=C?�=�B�\��z����\�c�
CP�)                                    Bx�M�,  T          @��H�"�\��\)��=q�v=qCTu��"�\�)������ZG�Cb�                                    Bx�M��  "          @˅�#�
�33���H�d�C]޸�#�
�QG������Dp�Cg�                                    Bx�M�x  "          @�\)�"�\�   ����\\)C`�)�"�\�[������:�Ci��                                    Bx�M�  T          @�
=��Ϳ���G��~�C[�f����:=q���\�^\)Ch�R                                    Bx�M��  �          @��H��H�ٙ����
ffCW  ��H�5���b��Ce��                                    Bx�M�j  "          @ᙚ�\)������G���CR��\)�#�
����t\)Cd��                                    Bx�M�  �          @���Q�����=qC\��Q��Mp���  �tClff                                    Bx�M�  
�          @�ff��\)�$z��Ӆ{Cvuÿ�\)�s�
����a�C}��                                    Bx�M�\  
�          @�(���  �33��Cn�H��  �W
=��z��r��Cy�)                                    Bx�N  T          @���p��������#�CP5ÿ�p��(��ٙ�p�Cf��                                    Bx�N�  "          @�{�˅��G������CCY��˅��G���=qB�Cc�                                    Bx�N)N  
�          @�(�����=q��Q�CV�׿��1���\��CkQ�                                    Bx�N7�  T          @�(�����˅��R\)C]�����A���  �}
=CoaH                                    Bx�NF�  "          @�p����
��
=���� CZ���
�9����(�Cn��                                    Bx�NU@  "          A Q���ÿ�p���{\CV�����/\)��G�k�Cl\)                                    Bx�Nc�  �          @���*=q�\)��\)k�C?�3�*=q������
=��\CX.                                    Bx�Nr�  T          @������?��
�
=q���HC"�H����?�녿c�
�{C$0�                                    Bx�N�2  �          @�p�����?���	����(�C)������?z���
��\)C.!H                                    Bx�N��  T          @�
=���>��
����  C0����녽�G��������C5
                                    Bx�N�~  �          @�=q���\�\)�Q���=qC9�R���\���
�{��
=C>s3                                    Bx�N�$  �          @����ff?�\)��=q�,�
C$����ff>�Q���  �4�
C/޸                                    Bx�N��  �          @߮��?���tz����C&J=��?z��������C.0�                                    Bx�N�p  �          @�ff���@��p  �{C�����?�(����H�%�C$Ǯ                                    Bx�N�  
�          @��
�R�\��\��z��vG�C<Ǯ�R�\��G�����g  CP!H                                    Bx�N�  �          @��
������\)��{�Y�C|�����������{�+�HC�&f                                    Bx�N�b  
�          @�׿�p��Q������y
=Cys3��p����\���\�K�\C~�q                                    Bx�O  �          @��
�h���ff��\)�Cv�f�h���`  ��z��r  CxR                                    Bx�O�  T          @��<���33��\)�t  CV�R�<���Vff����S��Cd�
                                    Bx�O"T  �          @�G��q녿5��
=�l�C>�f�q���\���\��CPaH                                    Bx�O0�  �          @��H�\)�����=q(�CJ��\)�   ��{�x��Ca�                                    Bx�O?�  T          @�{�7
=?\)��=q�RC(�)�7
=�aG���G�k�CE�                                    Bx�ONF  T          @��AG�?:�H�����|�HC&n�AG��z���p��~
=C>�                                    Bx�O\�  �          @�p��k��5�����h��C>�3�k���(���Q��X\)CP:�                                    Bx�Ok�  T          @�\)�>{�   ���Q�C=���>{��
=�љ��vp�CU                                    Bx�Oz8  T          @�\�E��p����� C:ٚ�E��=q��p��v��CR��                                    Bx�O��  �          @����qG������Q��r�C;
�qG���33��  �d{CN�q                                    Bx�O��  �          A���  ����(��^�C9�H��  ��\)���
�R\)CJ�\                                    Bx�O�*  �          AG����R?�
=���
�^�C":����R���
����fG�C5                                    Bx�O��  �          A�����?k���(��m��C'T{����!G�����offC<�                                     Bx�O�v  T          A
=��
=@��=q�K(�C���
=?.{��(��Y(�C+�{                                    Bx�O�  �          A (��B�\������y��CV\)�B�\�c�
��ff�X  Ce�                                     Bx�O��  �          Aff�@  ����\#�CT���@  �^{�׮�^��Ce.                                    Bx�O�h  
�          A�
�`�׿+����p�C>�\�`���G���\�m�RCT�)                                    Bx�O�  �          A
=�u=�G���\)�x{C2aH�u��p���\�o\)CI�                                    Bx�P�  T          A33��Q�?(����G��o�HC*����Q�fff��Q��nffC@��                                    Bx�PZ  �          A(���{@Z�H��Q��I=qCc���{?������H�c
=CaH                                    Bx�P*   "          A�
���?B�\��(��P33C+������:�H��(��PQ�C<{                                    Bx�P8�  �          A
=���>Ǯ���iffC/  ������H��\�d�CCc�                                    Bx�PGL  �          A  ����=p���33�ZffC<n����33��Q��K�\CL��                                    Bx�PU�  �          A���ff>������U�\C1
=��ff���R�љ��P�CB�                                    Bx�Pd�  �          A �����\?��
��G��B�C#:����\>\)��\)�K
=C2aH                                    Bx�Ps>  
�          @��R����@Mp���z��z�C�����@����  �'ffCh�                                    Bx�P��  �          @��
����@�����33�Tz�C	ff����@w���  �@z�C
��                                    Bx�P��  �          @�\)���@�\)@�{B2
=B����@�@g
=B��B�{                                    Bx�P�0  T          @�{�,��@�ff@�33B  B�q�,��@��@J�HA�\)B�Ǯ                                    Bx�P��  �          @�\�3�
@�G�@���B$G�B��3�
@�
=@`  A���B�W
                                    Bx�P�|  �          @���Dz�@�G�@�Q�B�HB�8R�Dz�@�=q@,��A��B�                                    Bx�P�"  �          A\)�\)@��?��HA$(�B�B��\)@��
��33��B�                                    Bx�P��  �          A���@��þ�=q����B�aH��@�Q��p��iG�B׊=                                    Bx�P�n  "          @������@�\��(��K�B�����@����
=q���B��f                                    Bx�P�  �          @񙚿У�@�>�ff@]p�B�aH�У�@��Ϳ�p����Bͳ3                                    Bx�Q�  �          @ۅ��
=@y���o\)�
=C	@ ��
=@;���G��%�C33                                    Bx�Q`  �          @�  �p  @R�\�����RC
�R�p  @  ��Q��=�\C                                    Bx�Q#  �          @�������@
�H�Q���33C������?�\)�%�����C!J=                                    Bx�Q1�  �          @�����?�{����3��C$������=�G�����;�HC2k�                                    Bx�Q@R  T          @��H�E��
=�ٙ�{CH��E�/\)�ʏ\�c�\C]��                                    Bx�QN�  �          @ᙚ�`      ��{�n{C4  �`  ��������c��CI��                                    Bx�Q]�  T          @�{��?Ǯ��  �.��C����?���  �;�RC,�3                                    Bx�QlD  T          @ʏ\���?����a��33C'� ���>�=q�l�����C1                                    Bx�Qz�  �          @�\���R>�(���  �@{C.���R�Q����R�=��C=�                                    Bx�Q��  
�          @�\)���H?������=��C,����H�.{�����==qC;�3                                    Bx�Q�6  "          @������?������=�
C'��녾�����R�BG�C7#�                                    Bx�Q��  
�          @�����?�����(��7(�C&���������  �<��C5�                                     Bx�Q��  "          @޸R����?�z���ff�+  C �3����>��H���R�7
=C.8R                                    Bx�Q�(  "          @��H��  ?��������1{C%ٚ��  <��
��p��8�C3�\                                    Bx�Q��  T          @ָR���
?�G����R�?ffC'����
�aG������DffC6�{                                    Bx�Q�t  �          @���QG�@��
���H�C�
B�8R�QG�@�33�5��£�B��f                                    Bx�Q�  T          @�
=����@�Q��QG���z�C������@e��Q��p�C(�                                    Bx�Q��  
Z          @޸R����@h���p���33CaH����@'������#
=C�                                    Bx�Rf  T          @����
=@#�
��Q���\C����
=?�����  �/��C"�3                                    Bx�R  �          @�ff��G�@����^{���C�R��G�@K����
�G�C�                                     Bx�R*�  T          @�\)����@�=q�   ��ffC�����@�p��K����HC�q                                    Bx�R9X  "          @�z��\)@�ff�2�\���HC^��\)@w��w
=�{C�                                    Bx�RG�  
�          @ָR�qG�@X������!z�C
��qG�@{��Q��Cp�C��                                    Bx�RV�  T          @�Q��W
=@ff����Q33C��W
=?xQ���\)�jG�C#�3                                    Bx�ReJ  �          @ٙ��S�
@'
=��p��K=qC�S�
?�  �����g�HC^�                                    Bx�Rs�  �          @Ӆ�QG�>Ǯ���R�p�C-:��QG������(��jz�CE��                                    Bx�R��  
�          @ڏ\�^{?^�R�����i  C%�
�^{�
=����kQ�C=��                                    Bx�R�<  �          @��H�aG�?�����H�^�C�3�aG�>�Q����
�o��C..                                    Bx�R��  �          @����\��@5���
=�E�HC���\��?�
=�����d33Cp�                                    Bx�R��  T          @����aG�@,(����R�K�C���aG�?�p����R�h
=C �                                    Bx�R�.  �          @�p��j=q@G
=���\�?�HC���j=q?���{�_�\C��                                    Bx�R��  
�          @�(��o\)@j�H���H�*�HC� �o\)@�����Op�C�                                    Bx�R�z  �          @���r�\@Dz���\)�;�HC
=�r�\?�����\�Z��C��                                    Bx�R�   �          @��e�@n�R��ff�.��Cٚ�e�@ff��\)�T��C�R                                    Bx�R��  "          @��q�@Fff����;�HC�f�q�?�z���33�[(�C@                                     Bx�Sl  �          @�G��z�H@Fff�����:{C�f�z�H?�33��(��X��C#�                                    Bx�S  �          @�\���R@Dz���(��2{C�f���R?�z���  �O  C��                                    Bx�S#�  �          @��H����@G������.33C�����?�(����KQ�C.                                    Bx�S2^  "          @����@Z=q��p��,��C�)���?�(����
�Lz�Cs3                                    Bx�SA  T          @�\��(�@�(���\)��RC���(�@8�������6p�C��                                    Bx�SO�  �          @ָR�e@����%���B�\�e@�{�s33�
33C�)                                    Bx�S^P  �          @��H�i��@����ff���RB�
=�i��@�p��[���Q�B��                                    Bx�Sl�  �          @أ��hQ�@����C�
��=qC �R�hQ�@e���z���Ck�                                    Bx�S{�  �          @�ff�|(�@h�����
��C	G��|(�@Q�����A(�C�                                    Bx�S�B  
�          @Ӆ�s33@��H�c33���C�)�s33@AG���  �+=qC}q                                    Bx�S��  �          @�=q�j=q@��R�!G���z�CaH�j=q@i���fff�p�C#�                                    Bx�S��  �          @�(��`��@�Q�� �����
C���`��@\���b�\�
=C��                                    Bx�S�4  �          @�z��c�
@�=q�3�
�ۅC#��c�
@L(��q��z�C
#�                                    Bx�S��  T          @�  �G
=@�ff�����RB��f�G
=@i���aG��
=Cn                                    Bx�SӀ  �          @���G�@`  �`���(�C���G�@������=�C��                                    Bx�S�&  
�          @�p��U�@u��@�����C��U�@9���z�H�&\)C                                    Bx�S��  �          @�{�e�@^{�Mp��
=C�{�e�@\)�����*��C)                                    Bx�S�r  �          @�Q��p  @g
=�.{�ܸRC\�p  @0  �e��C��                                    Bx�T  "          @�33�n�R@y���0���ظRC�3�n�R@AG��l����C�q                                    Bx�T�  �          @�p��}p�@fff�AG���33C	���}p�@*=q�w��=qC!H                                    Bx�T+d  �          @�G�����@S�
�E�����CQ�����@
=�vff��C(�                                    Bx�T:
  �          @�G���p�@�Y���Q�C����p�?����u� ��C&                                    Bx�TH�  �          @�33�W�    ��G��a  C3���W���{����T�RCJ                                    Bx�TWV  T          @�{�HQ�?!G���  �j�RC(���HQ�O\)��
=�h�CB��                                    Bx�Te�  �          @�z�����?�33����)�HC!�)����>k����H�5��C0�                                    Bx�Tt�  T          @������@{�c33�z�C������?�z����H�&
=C".                                    Bx�T�H  �          @ʏ\��p�@1G��u����Cn��p�?�\)��{�3�C�\                                    Bx�T��  �          @���:=q@AG���33�D�C��:=q?�=q��  �k��C��                                    Bx�T��  �          @�G��'
=@"�\���V
=C���'
=?����{�y��CY�                                    Bx�T�:  �          @Ǯ��(�@ ����{�)G�C
=��(�?G����\�=�C)J=                                    Bx�T��  T          @ƸR���H@�\��G��#�C�R���H?�������;�C%�                                    Bx�T̆  �          @�z��y��?�Q���G��2{C�{�y��?.{�����FffC*&f                                    Bx�T�,  �          @���\)@  �|���#
=C���\)?�����p��;C$��                                    Bx�T��  �          @���(�?�(��u�z�C�\��(�?Q���\)�.
=C)Y�                                    Bx�T�x  T          @�p����H?�������)z�C"@ ���H>8Q���(��4��C1��                                    Bx�U  T          @����z�?5���H�*�HC+B���z������,(�C:u�                                    Bx�U�  "          @�z���=q?
=q��  �7z�C-@ ��=q�O\)���R�5�\C>
=                                    Bx�U$j  T          @�(����>�=q�����,��C0����녿�G���p��'�HC?��                                    Bx�U3  �          @Ӆ����G����\�$��C5O\����=q���
��\CC
=                                    Bx�UA�  �          @�=q��33��ff��p��5\)C9�=��33��  ����&(�CH��                                    Bx�UP\  �          @�\��G�=�����
=�233C2����G������G��*\)CB�{                                    Bx�U_  �          @�{���?5��
=�B�
C*����녿:�H��
=�B�RC=��                                    Bx�Um�  �          @�33�|��?����z��P�RC ���|�;L������ZffC6�                                    Bx�U|N  �          @�  ���H�u���H�7�RC?�\���H�33���
�!��CN�)                                    Bx�U��  
�          @�������B�\�����I\)C=������G����R�3��CO0�                                    Bx�U��  
�          @�Q���
=��=q��(��H�HC7����
=��ff��=q�9�CI�                                    Bx�U�@  "          @�z����>����Q��2�RC1
=��녿�Q���(��,�CA0�                                    Bx�U��  T          @ᙚ��Q�\��{�&p�C8)��Q���H��(��p�CE�q                                    Bx�UŌ  �          @�Q���=q��Q��\)�Q�CH�H��=q�@  �QG����CR�H                                    Bx�U�2  T          @�33��ff�'
=�]p���  CO�{��ff�`  �#�
��33CWG�                                    Bx�U��  �          @�ff���R����vff�{CM+����R�]p��>�R�˙�CU��                                    Bx�U�~  T          @����
���|���
��CG����
�>�R�N{��p�CQ�                                    Bx�V $  T          @�z���33��p�����CCu���33�&ff�]p���G�CM��                                    Bx�V�  �          @�����zῡG����H���CA!H��z�����c33��Q�CK��                                    Bx�Vp  �          @��
�0  @���xQ���HB�u��0  @^{����Dp�C ^�                                    Bx�V,  T          @ڏ\�.�R@���z���B���.�R@QG����\�M�\C�                                    Bx�V:�  "          @��5�@��H�qG��(�B�G��5�@B�\���
�F=qC�R                                    Bx�VIb  "          @�p��'�@�Q��n{��HB�8R�'�@N{��(��Fz�C�                                    Bx�VX  "          @���(�@���U��B����(�@N�R����D  B�B�                                    Bx�Vf�  �          @��H�У�@�33�,�����
B�  �У�@fff�|(��3B�R                                    Bx�VuT  �          @�
=�<(�@I�������3{C��<(�?޸R��Q��^�HCc�                                    Bx�V��  T          @��H�h��@:�H����){C=q�h��?��
����L��C+�                                    Bx�V��  �          @�(���Q�@'������%\)Cu���Q�?�Q���\)�Az�C$ff                                    Bx�V�F  �          @�G���Q�?�p���G��0{C!H��Q�?���p��C�\C,޸                                    Bx�V��  �          @�=q��=q?.{���
�@�C+���=q�O\)����?�C>�{                                    Bx�V��  �          @�33��  >Ǯ�����F�C.�q��  ���������@�\CB�H                                    Bx�V�8  T          @�{��(�?�=q��33�6(�C&+���(���33���R�;�C8��                                    Bx�V��  T          @�p���\)?�p����H�*p�C!�R��\)=�G����H�6��C2�H                                    Bx�V�  �          @ȣ���p�>������/Q�C-����p��Tz����R�,�\C>�H                                    Bx�V�*  �          @�G���녿�������?�RCB�������H���
�$�CR                                    Bx�W�  �          @Ϯ�x�ÿ�(���{�G��CH��x���9�������%Q�CX�f                                    Bx�Wv  �          @��H�u��(���{�P��CE�{�u�/\)���\�0�\CWn                                    Bx�W%  T          @��
�`  �!G���G��e=qC>5��`  ��\���H�JffCU!H                                    Bx�W3�  �          @�\)�g�>\)����_�
C1�=�g���  �����R��CJ��                                    Bx�WBh  �          @�ff�S33=�G����\�q�
C2��S33�����\�a�CN�
                                    Bx�WQ  �          @�\)�O\)>���z��t�\C1���O\)��
=�����dz�COW
                                    Bx�W_�  �          @����|(�>����  �X��C.��|(���ff����Q�CF5�                                    Bx�WnZ  
�          @�ff���?���p��=��C-ff������\���H�9�HC@Ǯ                                    Bx�W}   
�          @�33��ff?�����
=��C'#���ff�aG�����%z�C6�{                                    Bx�W��  "          @�G����=�Q������==qC2�\���������\�2�
CExR                                    Bx�W�L  �          @˅��(�?L����Q��5�\C)����(��#�
��G��6�HC<B�                                    Bx�W��  �          @��H���=�\)�u��C3.�����\)�j=q�{C@�{                                    Bx�W��  �          @�����R�   �]p����C9� ���R��ff�HQ��CD��                                    Bx�W�>  �          @����
>\�q��ffC/�����
�Q��mp��G�C=�                                    Bx�W��  T          @˅����?E���33�!(�C*�f���ÿ����z��"C:�{                                    Bx�W�  T          @�Q���(�?Y�����
�1��C(�q��(��\)��p��4=qC;@                                     Bx�W�0  T          @�{�~{>\��{�Fz�C.���~{������=q�?�CC��                                    Bx�X �  
�          @�\)�U>�  ����e��C/�U��z������Y\)CJ�
                                    Bx�X|  
�          @�
=�n{?(�����Q�HC*���n{�u����N�\CBu�                                    Bx�X"  T          @�  �\��@�
��Q��G{C&f�\��>�����_�C,c�                                    Bx�X,�  �          @�=q�'
=@7
=��ff�O��CJ=�'
=?�z���33�|(�C�                                    Bx�X;n  "          @�=q��\)�B�\��p��%�\C6J=��\)��  �y�����CE��                                    Bx�XJ  
�          @�=q���ͼ��~�R�Q�C4W
���Ϳ���qG���CB�R                                    Bx�XX�  "          @��H��  ���
���%�C7�H��  �У��w
=�=qCF�3                                    Bx�Xg`  T          @ə����R�E����H�"Q�C=E���R���hQ��33CKJ=                                    Bx�Xv  T          @�{���H�u�����"�C?�����H�(��`���
ffCM��                                    Bx�X��  �          @�p�����?B�\�\)� ��C*����Ϳ�������"�C:��                                    Bx�X�R  
�          @�����\?��p  �=qC,�H���\�(���o\)��\C<&f                                    Bx�X��  
(          @����=q=�G��[���C2�f��=q�}p��Q���C?�)                                    Bx�X��  
�          @����ff>��H�XQ��G�C.���ff��R�Vff�=qC;�                                    Bx�X�D  
�          @�����Q�?�(��QG��
p�C"���Q�>�{�dz���C/�                                     Bx�X��  �          @�{��(�?����Dz��z�C'���(�<��
�P  �G�C3��                                    Bx�Xܐ  "          @��H��  ?����9����  C ff��  ?z��P�����C,��                                    Bx�X�6  
�          @����
@G��6ff���\Cٚ���
?}p��Vff�=qC&s3                                    Bx�X��  �          @�  �Z�H@8���L(����C���Z�H?ٙ��|���7�C��                                    Bx�Y�  T          @�
=��
=?������=qC���
=?z�H�:=q��\C&޸                                    Bx�Y(  �          @����\)@G��!G���C�=��\)?�=q�G���C"��                                    Bx�Y%�  �          @�����\@�
������
C�f���\?����8���\)C!�                                     Bx�Y4t  T          @�{��p�@\)�z���G�Cs3��p�?����;���(�C#ٚ                                    Bx�YC  T          @�(���{?�  ���R���C����{?�G������z�C'��                                    Bx�YQ�  �          @�(���\)?�33��=q����C&f��\)?�Q��ff��G�C&��                                    Bx�Y`f  �          @�33����?�׿�  ���\C�=����?��
��\����C%                                      Bx�Yo  T          @�\)��z�?�z������33C�f��z�?���
=��\)C%:�                                    Bx�Y}�  �          @�����?��R���
����C0����?����ff���\C#�=                                    Bx�Y�X  T          @����?��ÿ��H��G�Cu����?��R��p���(�C$�q                                    Bx�Y��  �          @������?�=q��33��\)C!
���?s33����ffC(ff                                    Bx�Y��  �          @��\����?��������Q�C$�����?��G���  C-8R                                    Bx�Y�J  �          @���  ?�=q������C�H��  ?�
=�	���ƣ�C%J=                                    Bx�Y��  �          @����n{@p��˅���Cz��n{?���z���RC=q                                    Bx�YՖ  T          @����H��@e��{���HC0��H��@6ff�-p��G�C	�=                                    Bx�Y�<  �          @��\�n{@�
��\���C��n{?�  �,�����C
=                                    Bx�Y��  �          @����n�R@�
���H��\)C!H�n�R?�\)�33����C��                                    Bx�Z�  �          @��R�k�@7
=�#�
���\C5��k�@�R�Ǯ���HC{                                    Bx�Z.  �          @�{�k�@0  �n{�6=qC:��k�@녿����{C5�                                    Bx�Z�  �          @���l(�?�{�����33C8R�l(�?��ÿ���C Q�                                    Bx�Z-z  �          @��\�x��?���ff��ffC��x��?��H����(�C"�3                                    Bx�Z<   �          @���\)@�
��  ��G�C��\)?�Q��ff���HC 33                                    Bx�ZJ�  �          @�
=�{�?u��33��G�C&5��{�>�=q�
=���
C0�                                    Bx�ZYl  �          @�z��w�?���p���G�C,c��w��8Q���\��{C6�R                                    Bx�Zh  �          @�33�u?   ���R��\)C,���u�L����\�߅C7
=                                    Bx�Zv�  �          @�(��w�?녿�Q���\)C+��w����G���ffC5�H                                    Bx�Z�^  
�          @�  �HQ�?У׿
=�(�Cp��HQ�?����{���RC��                                    Bx�Z�  �          @�p��h��>.{����
=C1^��h�ÿ���(�����C<Y�                                    Bx�Z��  
�          @����l��?Tz��33��{C'h��l��>W
=��=q��z�C0�                                    Bx�Z�P  �          @�=q�z=q?Tz��\)��=qC'�q�z=q>k���ff���
C0��                                    Bx�Z��  T          @����n�R?�Ϳ���p�C+���n�R�u��G����C4�f                                    Bx�ZΜ  T          @��H�r�\>�=q���H���RC/���r�\��  ��(���
=C7�\                                    Bx�Z�B  
�          @�p��o\)?�z�L���2=qC{�o\)?��
������Q�C!{                                    Bx�Z��  �          @z=q�W�?�녿��\)CǮ�W�?��ÿ�Q����C��                                    Bx�Z��  T          @|���P  @
�H����
=C5��P  ?�׿�
=���C��                                    Bx�[	4  �          @z=q�Z=q?���\)�33C^��Z=q?�p�������Cz�                                    Bx�[�  �          @~�R�^�R?�(���
=��C� �^�R?�(��}p��jffC(�                                    Bx�[&�  �          @}p��z=q>�\)��p����\C/��z=q>\)��G����C2                                    Bx�[5&  �          @|���z�H>�p��L���<��C.���z�H>�=q���
��(�C/�q                                    Bx�[C�  T          @\)�~{>�{�u�^�RC/��~{>����.{� ��C/�R                                    Bx�[Rr  �          @\)�|(�?(�>�?�{C+33�|(�?�R��Q쿧�C+�                                    Bx�[a  �          @��H����?
=q�����C,^�����>�׾�����{C-h�                                    Bx�[o�  �          @������?8Q�8Q��{C)�
����?�R��
=���HC+Q�                                    Bx�[~d  
�          @�������?xQ�>k�@P  C&z�����?}p����У�C&0�                                    Bx�[�
  �          @���{�?�=q>�=q@qG�C!Y��{�?���W
=�7�C!0�                                    Bx�[��  "          @�
=�vff?�Q�>�@ҏ\C@ �vff?�G��\)���CxR                                    Bx�[�V  T          @���l��?�
=?8Q�A  C}q�l��@�
    ��\)C�                                    Bx�[��  
Z          @�
=�X��@�?�G�A�\)C=q�X��@�R>�
=@�p�C�q                                    Bx�[Ǣ  T          @�\)�\��?�p�?�A��\C5��\��@ff?!G�A��CǮ                                    Bx�[�H  
�          @��
�l��?޸R>��
@�CǮ�l��?޸R���
��{CǮ                                    Bx�[��  
�          @�{�j=q?�\)?�@�G�C�q�j=q?�
=�.{��
C.                                    Bx�[�  "          @�Q��l(�?��?uAP(�C���l(�@ff>k�@ECY�                                    Bx�\:  T          @���s33@ ��?@  A33C.�s33@�ü#�
��\)C�
                                    Bx�\�  T          @����tz�@ff?��@�z�C+��tz�@
=q�u�Dz�C��                                    Bx�\�  �          @��\�c�
@�
?O\)A-G�C
=�c�
@�ͽu�O\)C�                                     Bx�\.,  �          @�  �X��@=q?aG�A?
=C���X��@$z�#�
���HC�{                                    Bx�\<�  �          @�
=�N{@#33?uAQp�C���N{@.{���
�W
=C�=                                    Bx�\Kx  T          @����c�
@#�
>��@ÅCJ=�c�
@#�
�����p�CL�                                    Bx�\Z  
�          @��Q�@3�
?p��AD��Cz��Q�@=p�����=qC	�q                                    Bx�\h�  T          @�p��Z�H@0��>��H@��HC!H�Z�H@0  ����z�C33                                    Bx�\wj  �          @��R�^{@333�L�Ϳ��C��^{@&ff��ff�YG�C(�                                    Bx�\�  �          @��
�S33@O\)=�Q�?���C���S33@C33��=q�W�
C	33                                    Bx�\��  �          @����Fff@S�
<�>��C��Fff@Fff��z��l��C��                                    Bx�\�\  �          @�Q��j�H@녿333�Q�C\�j�H?�녿����(�C�                                    Bx�\�  
�          @���j�H@G��z���33CT{�j�H?�zῬ�����HC��                                    Bx�\��  �          @���g�@ff�
=��\)C�g�?�p������\)CT{                                    Bx�\�N  T          @���b�\@%������(�Cٚ�b�\@�׿������Ck�                                    Bx�\��  
�          @�{�g�@!G�����z�C.�g�@	����z����
C=q                                    Bx�\�  �          @��_\)@-p���\)�^�RC+��_\)@   ����Z�\C^�                                    Bx�\�@  �          @�ff�g�@!G������p�C:��g�@�ÿ�
=��{Cff                                    Bx�]	�  �          @����e�@
=��33����C���e�@zῗ
=�
=C�3                                    Bx�]�  �          @�{�i��?��H����z�C���i��?�{���\���
C5�                                    Bx�]'2  T          @��i��?��#�
�(�CJ=�i��?Ǯ�����{C�                                    Bx�]5�  �          @�p��~�R?��\)��RC@ �~�R?�G���Q��y�CE                                    Bx�]D~  	�          @�ff�l(�?��5�z�Cs3�l(�?��H������HCu�                                    Bx�]S$  �          @~�R�g�?Ǯ���R����C�f�g�?���\(��Ip�C�                                    Bx�]a�  
�          @�=q�r�\?�녾��
���\C�f�r�\?��O\)�8Q�C"�\                                    Bx�]pp  
�          @�\)��Q�?�(��   �ۅC#
��Q�?s33�h���Hz�C&�                                    Bx�]  �          @�=q�vff?�������C"���vff?s33�\(��D��C&!H                                    Bx�]��  "          @��
�vff?��
�#�
��HC!���vff?s33����o�C&)                                    Bx�]�b  "          @�
=��Q�?��׿
=�p�C$J=��Q�?Tz�xQ��TQ�C(J=                                    Bx�]�  
�          @�
=��Q�?���&ff��\C%33��Q�?@  ��  �[\)C)xR                                    Bx�]��  
�          @�
=��  ?�p��aG��@��C"����  ?���(���Q�C%.                                    Bx�]�T  
Z          @�  ��z�?\(��u�W
=C(33��z�?G��\���C)Q�                                    Bx�]��  
�          @������?=p��u�Y��C)�����?(�þ�����33C*��                                    Bx�]�  �          @�  ���
?u������C&����
?W
=����G�C(k�                                    Bx�]�F  T          @�  ���?xQ�#�
�(�C&���?W
=��\��p�C(z�                                    Bx�^�  �          @����(�?n{�.{��RC'E��(�?L�Ϳ   �أ�C(�R                                    Bx�^�  "          @��H��(�?��\�k��C�
C"�H��(�?���0���=qC%&f                                    Bx�^ 8  	�          @����=q?u����Q�C&��=q?5�L���/33C*
                                    Bx�^.�  "          @����ff?�\)��33��=qC%{��ff?h�ÿ=p��(�C'�=                                    Bx�^=�  �          @��\��z�?��Ǯ��\)C$G���z�?p�׿L���*�HC'G�                                    Bx�^L*  
�          @�����R?��
����Y��C##����R?���8Q��=qC%��                                    Bx�^Z�  T          @������?��׾�  �QG�C!33���?�Q�=p���C#��                                    Bx�^iv  
�          @�G����?��
������
=C"�����?���J=q�*{C%Y�                                    Bx�^x  �          @�z���\)?�\)���
���C%.��\)?�G����H��
=C&�
                                    Bx�^��  "          @�33��33?�
=���ǮC �\��33?��
�&ff���C"�                                    Bx�^�h  �          @����?������
��=qC � ���?����R��C"u�                                    Bx�^�  "          @�{��G�?���=�G�?�(�C%J=��G�?�����
���C%�=                                    Bx�^��  T          @��
���R?�33=L��?(�C$�����R?���Ǯ����C%��                                    Bx�^�Z  �          @�z���p�?��>�?�C"����p�?�G��\��p�C#.                                    Bx�^�   �          @�����?�Q�u�E�C$�����?�G��+��
ffC&�q                                    Bx�^ަ  �          @�����\?�=q�\���\C%����\?\(��B�\�(�C(�q                                    Bx�^�L  "          @�(�����?E����H���C)�����?��=p����C,�                                    Bx�^��  �          @��H��  ?W
=�.{�(�C(޸��  ?5�����
=C*�                                     Bx�_
�  �          @����{?u����Q�C'���{?aG��Ǯ��ffC(#�                                    Bx�_>  �          @��\���H?�z�=L��?&ffC!����H?��þ��H�ϮC"�                                    Bx�_'�  �          @��
��33?��R=u?J=qC�R��33?�33��\��\)C!{                                    Bx�_6�  
�          @���\)?�  >#�
@
=qCO\�\)?�
=��\�أ�C&f                                    Bx�_E0  �          @�{�|(�@�=L��?0��C��|(�?�33�:�H��RC:�                                    Bx�_S�  
�          @�33�u�?��R>��@`��C�\�u�?�Q��\���C#�                                    Bx�_b|  	�          @�����\?˅�k��=p�C�q���\?��׿O\)�+\)C!L�                                    Bx�_q"  
�          @�z���?��>�  @J=qC#���?����=q�Q�C#��                                    Bx�_�  
�          @������?�z�>�{@��C%������?�������ffC%z�                                    Bx�_�n  
�          @�=q��?}p�?�@ƸRC(���?�{=���?�(�C&�H                                    Bx�_�  �          @����(�?c�
?�\@�
=C)���(�?��\>�?���C'�=                                    Bx�_��  �          @��\��ff?}p�>\@�Q�C'n��ff?������p�C&�{                                    Bx�_�`  T          @�z���\)?z�H?(�A ��C&���\)?��>B�\@\)C$�f                                    Bx�_�  �          @��R��\)?c�
?�ffAZ�HC(.��\)?�(�?!G�A�\C#�f                                    Bx�_׬  
�          @�����?�
=?z�HAM��C$�����?�(�>�ff@�z�C }q                                    Bx�_�R  T          @�p���{?��?s33A;\)C&�{��{?��>��@�\)C#.                                    Bx�_��  �          @��
��p�?G�?�=qAX(�C*
=��p�?���?333A
�RC%��                                    Bx�`�  T          @�����?��?��A�{C,ٚ���?��\?��
AO
=C&�=                                    Bx�`D  �          @�Q�����>u?�=qA��C0�\����?=p�?��Ai��C*J=                                    Bx�` �  �          @�p���
=��\)?��A�C4���
=>�(�?��HA�C.�                                    Bx�`/�  �          @�=q����?�\?���A�G�C,�\����?��
?��Au��C%��                                    Bx�`>6  "          @�  �}p�?B�\?���A��\C).�}p�?��H?k�AHQ�C#�                                    Bx�`L�  �          @�Q��u?��?�A���C#ff�u?�{?^�RA<��CG�                                    Bx�`[�  �          @���s�
?�Q�?��A�{CQ��s�
?���?.{A{C#�                                    Bx�`j(  
�          @���s33?�\)?��RA��
C���s33?�(�?�\@ۅC�=                                    Bx�`x�  �          @�33�vff?�z�?�{Am�C���vff?�(�>\@���C��                                    Bx�`�t  �          @���s�
?���?Tz�A1�C}q�s�
@   =u?B�\CT{                                    Bx�`�  "          @��R�l��?ٙ�?xQ�AUG�CL��l��?�Q�>k�@ECJ=                                    Bx�`��  T          @�
=����?�p�?8Q�AG�C������?�\)�#�
��Q�C0�                                    Bx�`�f  
�          @�{���?�?(��A	��C�����?�ff�#�
��C!H                                    Bx�`�  
�          @�z��~{?��H?333A=qC�3�~{?��ͼ���33C�                                    Bx�`в  "          @���|(�?�?8Q�A�RC  �|(�?�p��L�Ϳ333Cff                                    Bx�`�X  
�          @�(����\?�z�?��\AL��C�q���\@	��>L��@(�C0�                                    Bx�`��  �          @��w�?��
?�Q�AzffCL��w�@
=>���@�ffCs3                                    Bx�`��  �          @����k�?�z�?�=qAi��C�=�k�@�>u@QG�CQ�                                    Bx�aJ  �          @����QG�@#33?}p�AW
=C{�QG�@.�R��Q쿗
=C�                                    Bx�a�  T          @�G��Tz�@�R?�G�A[33CE�Tz�@+��#�
�
=C&f                                    Bx�a(�  �          @�\)�W�@�?�p�A�{C��W�@\)>�=q@l��C�                                    Bx�a7<  �          @����R�\?��H?�\)A�  CB��R�\@"�\?s33AMCff                                    Bx�aE�  �          @�  �J=q@(�?�ffAȏ\C8R�J=q@.�R?J=qA+�C8R                                    Bx�aT�  �          @����G�@�H?�Q�A�\)C@ �G�@8��?��A ��C	=q                                    Bx�ac.  T          @��H�8��@\)?�z�A���C+��8��@6ff>��
@�ffC^�                                    Bx�aq�  
�          @�33�J=q@�R?8Q�A#�C���J=q@#�
���R���C�                                    Bx�a�z  �          @�����@,��?���A��C5����@@  >��@��C33                                    Bx�a�   D          @�33��@&ff@z�B
ffC&f��@S�
?���Az{B�Q�                                    Bx�a��  �          @�����@�@.{B&�C\���@AG�?�z�A��B�\                                    Bx�a�l  T          @��
�ff@
=@333B,�HC޸�ff@C33?޸RA���B�33                                    Bx�a�  
�          @����z�@{@(�B�C=q�z�@@  ?�{A�\)B�\)                                    Bx�aɸ  �          @~{���@  @�B{C	xR���@:=q?��Axz�C(�                                    Bx�a�^  "          @\)� ��@�?�ffAݙ�C	+�� ��@6ff?:�HA-G�Cc�                                    Bx�a�  �          @�p��޸R?�@K�BK\)C#׿޸R@A�@
=qA�B�q                                    Bx�a��  �          @����@\)@.{B%p�B�  ��@W
=?�G�A�(�B�\)                                    Bx�bP  
�          @�Q���@�R@(�B�\C�3��@AG�?���A��
B��                                    Bx�b�  
�          @����{?�z�@VffB]p�C5ÿ�{@7
=@�HBp�B���                                    Bx�b!�  �          @�����@/\)@�B\)B�  ��@\(�?�ffAp��B�{                                    Bx�b0B  �          @������@+�@*=qB!��B��
����@`��?�\)A�(�B�\                                    Bx�b>�  T          @�33����@�@S�
BU��B��\����@K�@�RB
=B���                                    Bx�bM�  "          @�(����\?�
=@e�B\)B�\���\@/\)@-p�B+\)B��f                                    Bx�b\4  "          @~{��z�?�p�@HQ�BV��B���z�@Dz�@�B�B�p�                                    Bx�bj�  T          @{���(�@��@5B:�B�Q쿜(�@Tz�?�33A�=qB�p�                                    Bx�by�  �          @~�R��(�@��@6ffB:B�  ��(�@U?�z�A�  B�33                                    Bx�b�&  T          @��H�z�@��?�(�Aϙ�CB��z�@&ff>��@��C }q                                    Bx�b��  
�          @����>�R?k��i���F�C"���>�R��R�mp��J�C?�\                                    Bx�b�r  
Z          @���5?ٙ��4z��!�C��5>�(��QG��B��C+xR                                    Bx�b�  
�          @�(��HQ�?�z��U��,��C#��HQ�>8Q��n{�GG�C0��                                    Bx�b¾  T          @�33�G�?��R�HQ�� �Cp��G�?��j=q�D��C)��                                    Bx�b�d  
�          @��\�Mp�?��AG��C.�Mp�?
=q�b�\�=p�C*k�                                    Bx�b�
  �          @�(��Vff?�z��Fff��\C���Vff>�=q�`  �8��C/T{                                    Bx�b�  	�          @�{�U?�z��C33��\CB��U?��c�
�9�
C+�                                    Bx�b�V  
�          @���O\)?�z��S33�,�C���O\)�#�
�e�?z�C4�3                                    Bx�c�  �          @����[�?����>�R���C��[�>��S�
�/�C1��                                    Bx�c�  �          @����Vff?��7���RC+��Vff?��W��3ffC+                                      Bx�c)H  T          @�(��S33@2�\������C� �S33?ٙ��C�
�{C��                                    Bx�c7�  
�          @��\�>{@333�>{�=qC���>{?�\)�vff�F�CJ=                                    Bx�cF�  �          @�  �`��@�\�8Q��\)C�`��?5�]p��0�C(�\                                    Bx�cU:  �          @��H�e?���+���C�\�e?���J�H�$p�C+:�                                    Bx�cc�  �          @�33�~{?��qG��#ffC��~{>����9�
C2)                                    Bx�cr�  T          @����@ff�\���z�C�
��?���Q��'�C-(�                                    Bx�c�,  
�          @�
=���@�
�aG��ffC�����>������*�HC-�                                    Bx�c��  "          @��R��33?�
=�e����C��33>��
�����+��C/�                                     Bx�c�x  
Z          @�\)��ff?ٙ��fff�ffC���ff=����~�R�'\)C2��                                    Bx�c�  "          @����G�?��
�fff�C!^���G��u�z=q�#
=C4�                                     Bx�c��  
�          @�������?�ff�g
=��C!+������#�
�{��#(�C4�=                                    Bx�c�j  �          @��H���
?�{�h����C �\���
�#�
�~�R�#
=C4�                                    Bx�c�  T          @�z���p�?�{�i�����C ����p��#�
�\)�"(�C4�                                    Bx�c�  T          @�(����?����hQ���C!
����#�
�~{�!�\C4(�                                    Bx�c�\  T          @��
����?��
�j�H�z�C!�
���ͽ����~{�"
=C50�                                    Bx�d  
Z          @��
���H?�(��h���=qC�����H=�Q������$C2�)                                    Bx�d�  T          @�(���ff?��
�`  �Q�CB���ff>W
=�z�H�(�C1�                                     Bx�d"N  �          @���\)?У��hQ���C �R��\)<#�
�~�R� ffC3�{                                    Bx�d0�  �          @�{����?�=q�h���
=C� ����>L����=q�$z�C1�
                                    Bx�d?�  �          @�ff���?���o\)�p�C����=�G���z��'�
C2�
                                    Bx�dN@  �          @���?�  �g
=��C����>���Q��"\)C2xR                                    Bx�d\�  T          @�{���@
=�a��z�Cff���>��H���H�%��C-�3                                    Bx�dk�  T          @�ff����@	���g
=��C������>��H��p��)C-��                                    Bx�dz2  T          @�
=���?�
=�g���Cs3���>�z���33�%
=C0n                                    Bx�d��  �          @��
���?����o\)�ffC0����>�  ���R�.��C0                                    Bx�d�~  �          @��
���H?�33�k��=qC :����H    ��G��%Q�C3��                                    Bx�d�$  T          @��
���
?�G��e����C+����
>\)�\)�#
=C25�                                    Bx�d��  T          @�z����
@��\(��p�C\)���
?���  �#
=C-xR                                    Bx�d�p  �          @Å��{@���dz��\)C����{?
=q����+��C-�                                    Bx�d�  T          @��
��{@(��fff�
=C����{?����,Q�C-(�                                    Bx�d�  
Z          @����\@{�`���
�C!H���\?
=����&�HC,��                                    Bx�d�b  "          @����G�@�\�]p��	
=C8R��G�?+���33�'
=C+��                                    Bx�d�  �          @�{��=q@�H�Y���G�C)��=q?Q����H�%�\C)޸                                    Bx�e�  �          @�z���  @3�
�Z�H���Cu���  ?������0�C$��                                    Bx�eT  "          @������@33�mp����C�=���?z����\�8C,                                      Bx�e)�  "          @�
=��Q�?�\��G��*�\C5���Q�#�
����>C4��                                    Bx�e8�  �          @�����p�?�z��y���!�Cs3��p�>\)���H�8p�C2{                                    Bx�eGF  �          @�������@�\�w���
C�
����>�=q���
�9��C0B�                                    Bx�eU�  "          @�����z�@�\�y��� �HCǮ��z�>�������:��C0h�                                    Bx�ed�  D          @�\)�xQ�@���G��*G�C�q�xQ�>u��G��E�
C0k�                                    Bx�es8             @��o\)@
=q�����,�C��o\)>��R���\�J��C/:�                                    Bx�e��  �          @��l(�@\)�����,��C�q�l(�>\����L�C.
                                    Bx�e��  �          @�ff�p  @���z��0C���p  >������K�C1�                                    Bx�e�*  �          @���tz�?��R����0  C���tz�=�������IG�C2p�                                    Bx�e��  T          @�  �y��@�
����*��C:��y��>L�������EQ�C1                                      Bx�e�v  D          @�  �z=q@{�}p��%p�Cn�z=q>��������C��C.5�                                    Bx�e�  2          @�����R@&ff�J�H��z�C�����R?����}p��#G�C&�                                    Bx�e��  
�          @��
����@ ���[��33CG�����?aG�����+�C(��                                    Bx�e�h  
�          @Å��\)@{�U��33C{��\)?c�
�����&\)C(��                                    Bx�e�  T          @��
��
=@&ff�QG��G�C�\��
=?��
�����%�
C'{                                    Bx�f�  T          @Å��ff@,���L(���=qC���ff?�33�����$�RC%�\                                    Bx�fZ  "          @����@!G��Q��G�CQ����?s33�����+
=C'k�                                    Bx�f#   
�          @��R���H@%�J�H� G�C+����H?���|���&
=C&O\                                    Bx�f1�  �          @�����@#�
�H���z�C�\���?���z�H�'�C&&f                                    Bx�f@L  "          @��H��G�@!��C�
��z�CxR��G�?���tz��#�RC&L�                                    Bx�fN�  �          @��H���@$z��Mp��  CO\���?��\�~�R�+�\C&8R                                    Bx�f]�  �          @��H��Q�@#33�C�
��p�C
=��Q�?���u�$�HC%��                                    Bx�fl>  "          @�����p�@'
=�AG���ffC����p�?����u��&(�C$�=                                    Bx�fz�  �          @������@'��:=q��{CQ����?�Q��n�R� ��C$^�                                    Bx�f��  T          @��\��@0  �@����{C����?�G��w��&33C#33                                    Bx�f�0  �          @�������@.�R�>{��C������?��\�u��"
=C#��                                    Bx�f��  �          @�\)����@.�R�@  ���RC�q����?�  �w
=�)p�C"�
                                    Bx�f�|  T          @�  �z=q@.�R�L(��{C
=�z=q?���G��2��C#G�                                    Bx�f�"  "          @��R��p�@-p��1G���33C���p�?�=q�i���=qC"T{                                    Bx�f��  "          @�(�����@(Q��7
=���C������?�(��l(��$�C#@                                     Bx�f�n  �          @����ff@>{�-p���Q�C� ��ff?˅�l(����CQ�                                    Bx�f�  T          @�����z�@1G��:�H���HC0���z�?����s33�$C"O\                                    Bx�f��  "          @�Q���@��HQ����C���?n{�vff�(
=C'�                                     Bx�g`  �          @��H��\)@$z��Fff� \)C����\)?���xQ��&�HC%�                                    Bx�g  �          @����\@$z��HQ����HCG����\?�ff�z=q�%(�C&\)                                    Bx�g*�  f          @�����{@4z��>�R���C����{?���xQ���C#33                                    Bx�g9R  �          @������R@3�
�<(�����C�q���R?����u���HC#&f                                    Bx�gG�  �          @������@0���;���p�C������?�ff�s�
��HC#��                                    Bx�gV�  �          @����Q�@0���<(�����C����Q�?�ff�tz���\C#��                                    Bx�geD  �          @������@/\)�3�
��=qC�����?�=q�l(��ffC#�\                                    Bx�gs�  �          @�G����@,(��5���G�C�3���?��
�l(���C$n                                    Bx�g��  �          @��H��p�@8���(Q����HC@ ��p�?���e��C!��                                    Bx�g�6  �          @�����@<���&ff�ǅC)���?�{�e��Q�C!33                                    Bx�g��  �          @��
��(�@:=q�.�R��{C޸��(�?\�l(��\)C!��                                    Bx�g��  �          @�z�����@<(��,�����C�3����?Ǯ�j�H�Q�C!p�                                    Bx�g�(  �          @����@9���/\)�ә�C.��?�G��l(���C"
                                    Bx�g��  �          @Å���\@'��C�
����CE���\?�\)�w��\)C&G�                                    Bx�g�t  �          @����\)@@���"�\����C� ��\)?�Q��c33�(�C G�                                    Bx�g�  
�          @�{��Q�@4z��1G��ԣ�CT{��Q�?�
=�l(��Q�C#Q�                                    Bx�g��  �          @�p�����@4z��*�H�ͮCh�����?�(��fff�{C"�                                    Bx�hf  �          @�����=q@333�%��
=C����=q?�p��`���Q�C"��                                    Bx�h  �          @���\)@3�
�5����CT{��\)?����o\)��C#��                                    Bx�h#�  �          @�  ��z�@#�
�S�
���C&f��z�?u��=q�"�C(J=                                    Bx�h2X  �          @�Q����@(���6ff�أ�C����?�p��l(��ffC%�                                    Bx�h@�  �          @Ǯ��{@�\����� ��CJ=��{>�
=����>�\C.8R                                    Bx�hO�  �          @�����{@����Q���RCG���{?���z��>��C,޸                                    Bx�h^J  �          @�����{@)���N�R��G�C}q��{?�=q������C&�q                                    Bx�hl�  �          @ȣ����@'
=�?\)��C���?���s33�33C&�=                                    Bx�h{�  �          @Ǯ��p�@!G��8Q���Q�C�
��p�?�{�j�H�(�C'Q�                                    Bx�h�<  �          @������@!G��>�R���C�����?����p  �  C'�q                                    Bx�h��  �          @�=q���R@(���9����Q�C�q���R?����n�R���C&Y�                                    Bx�h��  �          @�Q�����@,���=p���(�C������?��R�s�
�G�C%��                                    Bx�h�.  �          @ə���=q@(���Z�H���C
=��=q?}p����R�'��C'��                                    Bx�h��  �          @�����@&ff�\(��
=C�����?s33��ff�'�C(^�                                    Bx�h�z  �          @ʏ\���H@333�Tz���33C�
���H?�
=���%�C%��                                    Bx�h�   �          @�����\@-p��XQ���Cn���\?�����{�&��C&�
                                    Bx�h��  �          @�=q���@333�W�� ��Cu����?�z���
=�'�C%�                                     Bx�h�l  �          @�G����@,(��S�
��z�C�R���?�=q��(��$G�C&�                                    Bx�i  �          @��H���@�R�S33���C� ���?c�
������C)z�                                    Bx�i�  �          @ə���\)@$z��QG���Q�C� ��\)?}p���G����C(0�                                    Bx�i+^  �          @ə�����@��Q����HC)����?\(��\)�=qC)��                                    Bx�i:  �          @ə�����@{�O\)��  C�q����?h���~{��\C)8R                                    Bx�iH�  �          @������@��L(�����C(�����?Y���x���
=C*5�                                    Bx�iWP  �          @�  ��33@  �N{��=qC{��33?:�H�w
=��
C+�                                     Bx�ie�  T          @�ff���\@!��U����C#����\?n{���\�$\)C(��                                    Bx�it�  �          @Ǯ��\)@�R�N�R��33C\)��\)?k��}p���C(�                                    Bx�i�B  �          @Ǯ���@/\)�>�R���C����?�G��vff�\)C%
                                    Bx�i��  �          @Ǯ���R@333�>�R��(�CL����R?����w��=qC$W
                                    Bx�i��  �          @�ff��(�@1G��C33��C!H��(�?��\�z�H���C$�3                                    Bx�i�4  �          @�{���@,���S�
��C����?�����(��'(�C&8R                                    Bx�i��  �          @����G�@0  �G����C����G�?�(��~�R�!
=C%�                                    Bx�ì  �          @�z���\)@#33�>{��C�H��\)?����p����C&�H                                    Bx�i�&  �          @��H��\)@c33� ����
=C�)��\)@=q�S33��RC                                    Bx�i��  �          @���u@�{�+���Q�C���u@qG��(���33CxR                                    Bx�i�r  �          @�  ��G�@��
�У��|Q�Cn��G�@E��K�� 33C��                                    Bx�j  �          @����
=@{�������  C	���
=@4z��S33�
=CE                                    Bx�j�  �          @�ff���R@\(��z���z�C�����R@��aG��p�C�f                                    Bx�j$d  �          @��R�#�
@�33@�A��HB�=q�#�
@���<#�
=�Q�B���                                    Bx�j3
  �          @�G��0��@��H@(Q�Aأ�B��)�0��@�G�?(�@�=qB�aH                                    Bx�jA�  �          @��\�<��@�G�@ffA��B�  �<��@�Q�=L��>�B�
=                                    Bx�jPV  �          @�
=�R�\@��?B�\@�\B����R�\@�G���=q�MB���                                    Bx�j^�  �          @��R�\��@��\>�@�z�B����\��@��H�����t��B�                                    Bx�jm�  �          @��
�o\)@�{�@  ��(�C��o\)@o\)� �����C                                      Bx�j|H  �          @��\�`  @����z�H��
B���`  @o\)�0������C�                                    Bx�j��  �          @�33�n{@�����G�C�f�n{@G��O\)���C��                                    Bx�j��  �          @���q�@�33�޸R����C�R�q�@AG��QG��=qCff                                    Bx�j�:  �          @�  �z=q@�  ��{�x��C���z=q@N{�Mp���C��                                    Bx�j��  T          @����tz�@�녿޸R����C�=�tz�@Mp��W
=��RC�f                                    Bx�jņ  
�          @����g�@�
=��  ��
=C  �g�@W
=�Z�H�
��C	�                                    Bx�j�,  �          @����mp�@��
����{CT{�mp�@O\)�^{���C
�                                    Bx�j��  �          @�p��vff@�z��
=��{C��vff@E�N�R���C@                                     Bx�j�x  �          @�ff�}p�@�{����P��CY��}p�@Q��<(���  CaH                                    Bx�k   �          @�{��(�@������%G�C#���(�@S�
�)���ԸRC@                                     Bx�k�  �          @�(���  @��
��
=�:ffC���  @Q��1G���\)C��                                    Bx�kj  �          @����tz�@�ff��Q��d  CL��tz�@O\)�A����C��                                    Bx�k,  �          @����vff@��׿���J�RC\�vff@W��;���{C
�
                                    Bx�k:�  �          @������@�z�u��\C:�����@XQ��#�
���C��                                    Bx�kI\  �          @����H@��s33��
CY����H@[��%���=qC                                    Bx�kX  �          @������@x���
=q���C	�=����@*=q�c�
��RC)                                    Bx�kf�  T          @�p���(�@q��������C8R��(�@$z��`  �
=qC��                                    Bx�kuN  T          @�p���Q�@xQ��$z����C  ��Q�@\)�{��
=C0�                                    Bx�k��  �          @ƸR�s�
@�Q��1G����HC�=�s�
@!���p��)Q�Ch�                                    Bx�k��  �          @�=q�p��@��\�'��ĸRC�R�p��@8Q�����%(�C��                                    Bx�k�@  �          @�33�o\)@qG��_\)�(�C�R�o\)@G���
=�@
=C�\                                    Bx�k��  �          @�=q�e�@s33�b�\���C=q�e�@���G��EG�Ck�                                    Bx�k��  �          @�G��}p�@z=q�6ff����C\)�}p�@=q��{�(�C��                                    Bx�k�2  �          @�G���  @{��7��أ�C����  @�H��
=�(33C�\                                    Bx�k��  �          @�=q��ff@r�\�1��ѮC	����ff@����H�!��C��                                    Bx�k�~  �          @�=q�|(�@���3�
��\)C&f�|(�@#�
���R�'��C�                                    Bx�k�$  �          @�G��xQ�@����5�ׅC�R�xQ�@!G���\)�)C�q                                    Bx�l�  �          @�=q��33@W��E���C0���33?�=q���&33C#�                                    Bx�lp  �          @�=q�j=q@�z��,(��ɅC���j=q@:�H����(�Cc�                                    Bx�l%  �          @ə��P��@�G��8Q����B�L��P��@>�R��
=�5p�C	�                                    Bx�l3�  
�          @�G��P  @����7��ٮB�33�P  @?\)��
=�5\)C	n                                    Bx�lBb  �          @�  �\��@����"�\��ffB��R�\��@Fff�����'G�C

=                                    Bx�lQ  �          @����N{@��
�/\)���
B����N{@G���(��1�\C�                                    Bx�l_�  �          @�ff�C�
@���������RB��)�C�
@e��  �!�Cc�                                    Bx�lnT  �          @�
=�?\)@�녿�������B���?\)@u�u���B�                                    Bx�l|�  �          @�\)�HQ�@�p��
�H��33B����HQ�@g��~{�Cٚ                                    Bx�l��  �          @ƸR�O\)@�=q�����i��B�#��O\)@�  �^�R�G�C �R                                    Bx�l�F  �          @�p��>�R@�{��(��]�B����>�R@����\(���
B�k�                                    Bx�l��  E          @ƸR�3�
@�=q��  �`��B��3�
@�Q��aG��
��B��
                                    Bx�l��  �          @ƸR�:=q@�G���\)�LQ�B�=�:=q@�G��X����RB�=q                                    Bx�l�8  �          @�{�;�@�  ��
=�V�\B�Q��;�@�\)�[����B�p�                                    Bx�l��  �          @��
�Fff@��׿Ǯ�lQ�B�k��Fff@~{�\���	��C                                     Bx�l�  �          @�p��E@����\)�MG�B����E@�p��U���
B�
=                                    Bx�l�*  �          @��HQ�@�z῱��P  B�\�HQ�@����U�
=B���                                    Bx�m �  �          @����R�\@��R��  �b{B�(��R�\@|(��W��{C�f                                    Bx�mv  �          @ƸR�r�\@��������z�C}q�r�\@@���r�\�G�C��                                    Bx�m  �          @����x��@~�R�:=q���HCff�x��@{��Q��+  C��                                    Bx�m,�  �          @�Q��j�H@��ÿ���O\)B�
=�j�H@tz��L(���ffCٚ                                    Bx�m;h  �          @�  �w�@�{��p���ffC
=�w�@Q��e�Q�C��                                    Bx�mJ  �          @�\)�|(�@����{��p�C���|(�@AG��p  �=qC�                                    Bx�mX�  �          @�����
@��\�   ��Q�C�
���
@J=q�dz��

=C�                                     Bx�mgZ  �          @�33��Q�@��������G�Cc���Q�@0���tz���C�                                    Bx�mv   �          @��
��Q�@~{�$z���ffC	
=��Q�@'
=�|(��{C�                                    Bx�m��  �          @˅���
@�=q�(����  Cc����
@*�H��G���HC�                                    Bx�m�L  �          @ə�����@w��\)���C
������@)���fff�=qC�                                    Bx�m��  �          @��
��\)@s33��H��33C����\)@!G��o\)�(�C�f                                    Bx�m��  �          @�33���@y���(�����C
&f���@'
=�r�\�
=C�                                    Bx�m�>  
�          @�G����\@o\)�&ff��C!H���\@���xQ��z�C�                                    Bx�m��  �          @�G����H@c�
�5���Q�C�����H@��������CǮ                                    Bx�m܊  �          @ə���Q�@a��'��ģ�C����Q�@(��s�
��
C�                                    Bx�m�0  �          @ə����R@Z�H�=q����C  ���R@��dz��

=C&f                                    Bx�m��  �          @�=q���R@G
=����{C�����R?�z��Z=q��RC��                                    Bx�n|  �          @ȣ���{@HQ��{��z�C�f��{?��R�Q���33C{                                    Bx�n"  �          @ʏ\��{@_\)�=q��33C\)��{@  �fff�
�Ck�                                    Bx�n%�  �          @��H��p�@i���\)��  C���p�@p��`���z�C=q                                    Bx�n4n  �          @������@u��#�
���C
Q�����@   �w��\)C�\                                    Bx�nC  �          @����xQ�@vff�Fff���C5��xQ�@�
���
�0ffCB�                                    Bx�nQ�  �          @ə��a�@�33�L(���ffC�a�@   ����9�HC��                                    Bx�n``  
�          @�33�b�\@n{�l���
=C���b�\?��H���
�I(�C�                                    Bx�no  T          @��H�l��@|���P����(�C#��l��@�����8\)C�                                    Bx�n}�  �          @�=q��
=@l(��8Q���C
����
=@  ����"��C��                                    Bx�n�R  �          @�G��`��@x���^�R�z�C\�`��@����\)�CG�C�                                    Bx�n��  �          @˅�fff@}p��XQ��G�CG��fff@33��p��>(�C^�                                    Bx�n��  �          @����X��@�(��aG��\)Ck��X��@=q����E��C�)                                    Bx�n�D  �          @�z��Y��@��\�`���C���Y��@����\�Ez�C�                                    Bx�n��  �          @ȣ��Y��@�  �[���\Cc��Y��@���
=�C�C��                                    Bx�nՐ  �          @ə��\(�@��\�U� (�C(��\(�@(���p��?�RC�                                    Bx�n�6  �          @���^�R@�  �Y���C
=�^�R@��ff�@��C\                                    Bx�n��  �          @ʏ\�]p�@����[��Q�C�q�]p�@
=��\)�A��C�                                    Bx�o�  �          @��H�C�
@��j�H�33B�Q��C�
@=q��Q��P��C��                                    Bx�o(  �          @�33�G
=@�{�g
=�
�B�{�G
=@�����R�M��C�                                    Bx�o�  �          @�33�Fff@��\�o\)��B���Fff@�\��G��RG�C�\                                    Bx�o-t  �          @��H�Fff@�{�e�
G�B��Fff@����{�M\)C�f                                    Bx�o<  �          @���7
=@��p  ��B����7
=@Q����H�W(�C8R                                    Bx�oJ�  �          @�=q�3�
@�
=�q��{B�L��3�
@=q��(��X�
Cc�                                    Bx�oYf  T          @�=q�AG�@����l����B�#��AG�@Q������R33C�                                    Bx�oh  �          @ʏ\�N{@����aG��\)B����N{@����33�H�
C�R                                    Bx�ov�  �          @��H�QG�@����^{�\)C J=�QG�@{����FQ�C�R                                    Bx�o�X  T          @ȣ��`��@�33�K����
C���`��@"�\�����8C�                                    Bx�o��  �          @����X��@���Tz���
=C��X��@   �����?=qC��                                    Bx�o��  T          @ə��QG�@��XQ���B����QG�@#33����CQ�C\                                    Bx�o�J  T          @�Q��Vff@�ff�N{��33C �=�Vff@(Q����H�<�Cٚ                                    Bx�o��  
�          @Ǯ�c�
@�
=�:�H����C(��c�
@0����=q�/33C33                                    Bx�oΖ  �          @�\)�U�@�p��L(����HC ���U�@'
=�����<G�C�                                    Bx�o�<  �          @����s33@�(�������HC���s33@e��	������C�                                     Bx�o��  �          @�G���{@~�R=�?�
=C���{@mp����d(�C
xR                                    Bx�o��  �          @��\��
=@�  >aG�@(�C����
=@q녿���TQ�C
+�                                    Bx�p	.  �          @�z�����@�Q�\)��=qC�3����@i������  C��                                    Bx�p�  �          @�ff��@�z�L����\)C8R��@aG������HC�H                                    Bx�p&z  �          @��z�H@���{�UC#��z�H@U�7
=��\C��                                    Bx�p5   �          @�=q��G�@hQ��G�����C\)��G�@S�
��  �o
=C�                                    Bx�pC�  �          @�G����\@aG�>��?�G�Cz����\@S�
���H�@��C+�                                    Bx�pRl  T          @��H����@aG��u�
=C������@J=q�����yp�C�\                                    Bx�pa  T          @��
���H@z�H����{C	����H@\(���Q�����C��                                    Bx�po�  T          @�(���@tz������C8R��@Vff�����=qC�H                                    Bx�p~^  
(          @����=q@hQ�\�n�RC���=q@Mp��޸R��=qC�f                                    Bx�p�  �          @�z�����@n{��G�����C������@QG��������\C8R                                    Bx�p��  �          @�(���  @p  �����\C33��  @QG���z�����C                                      Bx�p�P  �          @�z�����@b�\�(���\)C� ����@B�\��z����
Cٚ                                    Bx�p��  T          @����ff@XQ�Y����CG���ff@333�z����
C.                                    Bx�pǜ  �          @���{@C�
��33�3
=C33��{@���{��z�C)                                    Bx�p�B  �          @�����{@?\)��z��5�CǮ��{@��p���=qC�q                                    Bx�p��  �          @�p�����@Y������QG�C�H����@(���!G���{CxR                                    Bx�p�  �          @�(�����@`  �����T��C8R����@.�R�%���{Cٚ                                    Bx�q4  	          @�z����\@S�
�������C)���\@���1G���RC޸                                    Bx�q�  �          @�����\@P�׿�z����C�����\@���0����G�CaH                                    Bx�q�  �          @����Q�@8��������HCaH��Q�?����K����C�q                                    Bx�q.&  
�          @�Q����@!G��#33���C�����?�\)�Tz���HC"�q                                    Bx�q<�  �          @�\)���H@>�R������{C  ���H@
=q�&ff��
=C�                                    Bx�qKr  �          @�
=���@7���33���CY����@33�'
=����CL�                                    Bx�qZ  �          @��R���@1G����
���CB����?�33�,(����CǮ                                    Bx�qh�  
�          @�Q���
=@)�����H��G�C����
=?�(��3�
��C�                                    Bx�qwd  T          @�{��\)@�������C����\)?�\)�5��C#�\                                    Bx�q�
  �          @������@&ff�
=q��{C#�����?����>�R��Q�C �                                     Bx�q��  "          @�33��z�@c�
���H�H(�C^���z�@6ff������
C�                                     Bx�q�V  �          @��
���@U��޸R���\CW
���@(��7
=����C��                                    Bx�q��  �          @�����(�@K��
�H����C\)��(�@	���L���
{C��                                    Bx�q��  �          @����
@AG��p��ͮC�R���
?���Z=q��CQ�                                    Bx�q�H  "          @����p  @^{����ȣ�C	5��p  @�`�����C�                                    Bx�q��  
�          @��R�b�\@j�H����33C�3�b�\@!��c�
��Cff                                    Bx�q�  
�          @����e@l���   �Σ�C)�e@!G��l(�� Q�C�R                                    Bx�q�:  �          @�ff�s�
@8���8Q���=qC�
�s�
?�\)�o\)�(\)C��                                    Bx�r	�  	�          @��\�z=q@*�H�>{� ��C���z=q?����p  �(�C ��                                    Bx�r�  "          @�z��x��@7��7�����C�=�x��?�{�n�R�&�CxR                                    Bx�r',  �          @��q�@HQ��7
=��{C\)�q�?�{�s�
�(�C��                                    Bx�r5�  T          @�z��`  @^{�-p���  C33�`  @\)�r�\�)��Cff                                    Bx�rDx  T          @�Q��qG�@1��5����C���qG�?�ff�j=q�'C��                                    Bx�rS  T          @����}p�@ff�C33�=qCT{�}p�?����l���(Q�C$�                                    Bx�ra�  T          @��
�n�R@��G���C���n�R?z�H�o\)�0p�C%T{                                    Bx�rpj  
�          @�  �c33@��G��{CL��c33?z�H�o\)�5C$��                                    Bx�r  �          @����mp�?�z��S�
�p�C��mp�>�{�l(��2��C.�=                                    Bx�r��  �          @��g�?���b�\�#��Cz��g�>���~�R�>{C,��                                    Bx�r�\  "          @�G��l(�?�
=�L(���
Ch��l(�?!G��j�H�1��C*:�                                    Bx�r�  �          @����u@G��O\)�C^��u?p���u�0��C&5�                                    Bx�r��  
�          @����p��@��_\)�{C���p��?333�����9�HC)ff                                    Bx�r�N  "          @�
=�p��?��a�� ��CG��p��>���|���8��C-                                    Bx�r��  �          @���~{@p��<��� �HC0��~{?�p��h���$�RC"�                                    Bx�r�  T          @��H�~{@G
=�Q���  C�q�~{@�\�U�(�C�)                                    Bx�r�@  �          @����\)@i������
=C	���\)@*�H�Mp��G�C+�                                    Bx�s�  �          @���\(�@��׿�\)���
C���\(�@J�H�=p���C	Y�                                    Bx�s�  T          @��R�ff@��Ϳ���yp�B�L��ff@fff�8Q��(�B�L�                                    Bx�s 2  T          @����Q�@�����H�X��B��Q�@k��-p����B���                                    Bx�s.�  
�          @��*=q@���.{����B�3�*=q@u��(��ə�B��                                    Bx�s=~  
(          @��\�*=q@�녽�Q�z�HB�\)�*=q@\)��\)����B�Q�                                    Bx�sL$  �          @���-p�@�=q�����Z�HB�{�-p�@�z�����(�B�W
                                    Bx�sZ�  
(          @�G��+�@��ÿE�� (�B�\�+�@�{����  B�8R                                    Bx�sip  
�          @����3�
@��\�O\)���B�u��3�
@�
=��R��z�B�\)                                    Bx�sx  "          @�(��&ff@�
=�������B�8R�&ff@�  �	�����B�                                    Bx�s��  �          @�Q��)��@�녿����ffB�aH�)��@mp��C�
�ffB�\                                    Bx�s�b  "          @��\�@  @�{�Ǯ��(�B�(��@  @e�A��C޸                                    Bx�s�  �          @�=q�;�@��R���R�yp�B��\�;�@h���>{�C �                                    Bx�s��  "          @�{����@�G���{�tz�B�녿���@��H�ff��Q�B�(�                                    Bx�s�T  "          @�����(�@�33�����p�B�\)��(�@��
�	����33B�W
                                    Bx�s��  �          @��ÿ�=q@����\���B�  ��=q@�  �����=qB��)                                    Bx�sޠ  "          @��H��\)@�녾L���{B�LͿ�\)@������\)B�8R                                    Bx�s�F  "          @��H�O\)@�
=��\)�J=qBƅ�O\)@�(������\B��)                                    Bx�s��  "          @���(Q�@n�R��G����B�B��(Q�@8Q��=p����CaH                                    Bx�t
�  T          @��
��
@������Y�B����
@\��� �����RB��\                                    Bx�t8  
�          @��H�G�@�=q���H�h��B�Q��G�@X���#33� =qB���                                    Bx�t'�  w          @���#�
@���=p���B��#�
@e�Q���=qB��)                                    Bx�t6�  
�          @��\��\@�
=���
�s�
B�
=��\@vff��\���\B�                                    Bx�tE*  
�          @�z��p�@��R>��?޸RB���p�@�\)��z���ffB�33                                    Bx�tS�  
�          @�����Q�@�Q�>��
@l��BոR��Q�@��\�����{�B�                                      Bx�tbv  �          @�(��Q�@��?5A Q�Bƽq�Q�@�����?�B���                                    Bx�tq  �          @�zῡG�@�33?k�A&ffB�
=��G�@���O\)�33B��                                    Bx�t�  T          @��H�B�\@���@   A�=qB���B�\@�{>��R@eBŏ\                                    Bx�t�h  
�          @��ÿ   @��R?�A��
B���   @��>#�
?�Q�B��H                                    Bx�t�  
�          @�=q���@�G��O\)��BҊ=���@�
=�����\B�p�                                    Bx�t��  T          @�z��  @�\)���
�g�
B�.��  @�  �5��  B�{                                    Bx�t�Z  T          @��Ϳ���@����(��ۅB�녿���@����{��Q�B��)                                    Bx�t�   �          @����G�@�
=������p�Bսq��G�@�G�����{B�Ǯ                                    Bx�tצ  
�          @��Ϳ�z�@�����
�uBٮ��z�@�녿�
=����B���                                    Bx�t�L  
�          @��H���H@�(�>��R@_\)B�Ǯ���H@�{��\)�}��B��
                                    Bx�t��  
�          @�(��   @��\�n{�(Q�B�#��   @\)�����B�33                                    Bx�u�  
�          @�����@��H���H�]G�B�  ���@k��'
=��G�B�33                                    Bx�u>  
�          @�ff�#33@��H���\�?�B���#33@`  �ff��B�33                                    Bx�u �  	.          @��
�AG�@n�R�B�\���C��AG�@P  ��p���z�C��                                    Bx�u/�  �          @����W�@W���=q�Lz�C�W�@333�	����\)C:�                                    Bx�u>0  "          @�  �333@tz´p�����B�� �333@Fff�+���
C�                                    Bx�uL�  
�          @�Q��8��@W��
�H��{C�)�8��@(��J�H��CǮ                                    Bx�u[|  T          @�����H@-p�?\(�A�ffBȊ=���H@5������BǞ�                                    Bx�uj"  "          @�?���@Z=q@c33B/  B�{?���@���@��A���B���                                    Bx�ux�  "          @�=q?�@S33@l��B033Bo=q?�@�33@Q�Aԏ\B�W
                                    Bx�u�n  	�          @�z�@'�@<(�@j=qB+��BAp�@'�@�  @{A���Bc33                                    Bx�u�  
(          @���?��@R�\@u�B5  Bp33?��@��
@ ��A�Q�B�.                                    Bx�u��  "          @��
?�(�@N�R@��BG��B��\?�(�@�p�@333A��B�
=                                    Bx�u�`  
�          @��H?�
=@_\)@w
=B9\)B��?�
=@�=q@�RA�B��                                    Bx�u�  �          @���?n{@p��@o\)B/��B�33?n{@�G�@�A�=qB���                                    Bx�uЬ  �          @�ff?#�
@b�\@��BA��B�\)?#�
@�{@*=qA���B�Q�                                    Bx�u�R  	�          @��?z�@x��@l��B,��B��?z�@���@��A��RB�ff                                    Bx�u��  
          @�{?5@p  @tz�B4�B���?5@���@�A�ffB�8R                                    Bx�u��  �          @�  ?�\@�33@g�B$��B��R?�\@��@z�A�G�B�k�                                    Bx�vD  �          @�
=>B�\@�p�@`  B   B���>B�\@��H?�Q�A���B��q                                    Bx�v�  �          @�  >�=q@���@XQ�B��B���>�=q@�p�?��A�z�B�                                    Bx�v(�  T          @�=q�8Q�@|(�@s33B/��B�=q�8Q�@�
=@33Aƣ�B�(�                                    Bx�v76  T          @�p�>�z�@�ff@I��BQ�B��>�z�@�
=?���Am�B��                                    Bx�vE�  �          @��R>�=q@�{@5�A�(�B���>�=q@��H?���A,��B�aH                                    Bx�vT�  �          @��.{@�G�@#33A֣�B�녾.{@�33?E�@�(�B��                                    Bx�vc(  �          @��H��{@��
@p�A�Q�B��3��{@��>�(�@��B��                                    Bx�vq�  �          @������@���@ ��A��\B�������@�z�>B�\?�(�B�.                                    Bx�v�t  �          @�Q�.{@��
@�
A�  B��׾.{@��>aG�@
�HB�ff                                    Bx�v�  �          @�{�!G�@�33?\A|��B�\)�!G�@�����\)�8��B��H                                    Bx�v��  �          @��\���@�{�������Bɞ����@��������\)B�aH                                    Bx�v�f  �          @��ÿ�ff@��H�xQ��"=qB�33��ff@���(Q���B��f                                    Bx�v�  �          @�{���@�33�
�H��
=Ḅ׿��@y���g
=�'(�B�ff                                    Bx�vɲ  �          @�ff�O\)@�ff�(Q����HB�z�O\)@g
=�\)�<��B�.                                    Bx�v�X  
�          @�=q���@����4z�����B��῱�@Z�H���
�@��B�G�                                    Bx�v��  T          @��R�Tz�@�G��
=����BǊ=�Tz�@s33�p���0\)B̊=                                    Bx�v��  �          @��þ�  @��?�{AE��B�����  @��Ϳ����z�B��                                    Bx�wJ  �          @��;�ff@���?n{A�
B��R��ff@����J=q��\B��                                    Bx�w�  �          @�����R@�
=?�ffA^�HB��f���R@����
=���RB��R                                    Bx�w!�  �          @�ff�G�@�G�?z�HA%G�BĨ��G�@��\�=p����
BĊ=                                    Bx�w0<  �          @��
�z�H@�p�?h��A��B�z�z�H@�{�B�\��B�aH                                    Bx�w>�  �          @��ÿ�=q@�=q?�Ay��B�8R��=q@��׾.{����B��)                                    Bx�wM�  �          @�=q���
@���?�z�Av{B֙����
@��H�B�\�ffB�\)                                    Bx�w\.  �          @��R��@�{?�\A�=qB�{��@�  >\)?�(�B�G�                                    Bx�wj�  �          @����  @�ff?�=qA?
=B�LͿ�  @�G�����ffB���                                    Bx�wyz  �          @�  ���R@�=q>�(�@���B�=q���R@�{��  �R=qB��H                                    Bx�w�   �          @�  �Y��@���?�z�A�G�Bƨ��Y��@�����
�W
=B�Ǯ                                    Bx�w��  T          @�G�<��
@���@\)A�p�B���<��
@�  ?�@�(�B���                                    Bx�w�l  �          @�33=�G�@���@�\A�
=B�W
=�G�@��?333@�(�B��{                                    Bx�w�  �          @�  ��\)@�{?�ffA0Q�B�z`\)@�Q�(����
B��                                    Bx�w¸  �          @�G���=q@��R?c�
A��Bծ��=q@�\)�B�\��(�BՊ=                                    Bx�w�^  �          @��R�У�@��
?.{@��HB���У�@��H�W
=�{B�(�                                    Bx�w�  �          @�G��%�@�\)�8Q���B��%�@�����
���
B���                                    Bx�w�  �          @����HQ�@��\��\)��{B���HQ�@X���2�\��p�C�3                                    Bx�w�P  T          @����>{@�p��z����\B��H�>{@U�O\)�{C��                                    Bx�x�  �          @�  �E�@��Ϳ�33��B�G��E�@W��Dz���RCs3                                    Bx�x�  �          @�{�AG�@��׿�Q��Lz�B��{�AG�@mp�����33C.                                    Bx�x)B  �          @���<��@��>�@�ffB���<��@��׿E���B��=                                    Bx�x7�  �          @���;�@�  >�Q�@�p�B�ff�;�@z�H�\(�� (�B�z�                                    Bx�xF�  �          @��R��  @'�?�=qAJ�RC�R��  @5�>k�@/\)C�                                     Bx�xU4  
�          @�
=�Q�@l��>���@���C�{�Q�@i���:�H�
=C�R                                    Bx�xc�  �          @�{�qG�@J=q>�  @7
=C{�qG�@E��5��C�R                                    Bx�xr�  �          @���=p�@��\>��H@���B����=p�@�G��B�\�	G�B��                                    Bx�x�&  �          @��H�7�@�\)>�
=@�
=B�B��7�@��Ϳp���"ffB�33                                    Bx�x��  �          @����)��@���>���@�(�B�R�)��@�ff�xQ��(��B�                                    Bx�x�r  �          @����#33@�Q�>W
=@��B�aH�#33@�����H�O
=B��                                    Bx�x�  �          @��\�*=q@��H��
=��\)B���*=q@�Q��G���=qB��H                                    Bx�x��  T          @���A�@���������B����A�@z=q�ٙ���  B�aH                                    Bx�x�d  �          @�p��X��@n{��
=�S�CY��X��@Mp���R��{C�)                                    Bx�x�
  �          @�  �e�@QG���\��=qC	���e�@!��:=q�\)C�R                                    Bx�x�  �          @�{��(�@C�
��  ��(�C� ��(�@=q�$z���C�q                                    Bx�x�V  �          @������@J=q��\)����CǮ���@�R�-p���=qC@                                     Bx�y�  �          @�z���ff@P  �������CB���ff@#�
�0����Q�C��                                    Bx�y�  �          @�\)���@U�33���RC�����@!G��J�H��C&f                                    Bx�y"H  �          @�=q�e�@�(��L���z�C���e�@n{���H��Q�C��                                    Bx�y0�  �          @�=q�k�@q녿��
���C=q�k�@Fff�333��\)C�H                                    Bx�y?�  �          @���j�H@]p�������C���j�H@,���C33�=qC�3                                    Bx�yN:  �          @�  �p  @`  ��p�����C�q�p  @1��9����33Cu�                                    Bx�y\�  �          @�G��`��@��
��
=�DQ�Cp��`��@g
=��
��\)C:�                                    Bx�yk�  �          @���a�@}p���{��\)C��a�@U��+���G�C�R                                    Bx�yz,  T          @��
��Q�@e���33��ffC
=q��Q�@=p��&ff�ܸRC�{                                    Bx�y��  �          @����g�@��H����,(�CxR�g�@hQ��
=q���C�                                    Bx�y�x  �          @�ff�Q�@��;u�p�B�aH�Q�@�z�Ǯ�}�B��                                    Bx�y�  �          @��R�L(�@�{>�33@a�B�ff�L(�@�33��  �!p�B�z�                                    Bx�y��  �          @���U@��R����33B����U@�(���G�����C �3                                    Bx�y�j  �          @��\�g�@\)�����a��CE�g�@\(���H����CxR                                    Bx�y�  �          @�\)��G�@u�����_�
C����G�@Q��=q�ƣ�C�f                                    Bx�y�  �          @�=q�w
=@��R��{�0  CxR�w
=@n�R�  ���C�                                    Bx�y�\  �          @���w
=@�{����S\)C���w
=@i���p��ƣ�C��                                    Bx�y�  �          @�=q�h��@��H��p����HC�f�h��@XQ��B�\��  C	#�                                    Bx�z�  T          @��\�j�H@�
=��  �p��C��j�H@hQ��'���33CG�                                    Bx�zN  �          @�
=�hQ�@�p���Q��h��C��hQ�@fff�"�\��
=C5�                                    Bx�z)�  �          @�(��e�@r�\������Ch��e�@C33�E���
C�\                                    Bx�z8�  �          @���`  @xQ���\���RC{�`  @J=q�A���C	��                                    Bx�zG@  �          @�ff�g
=@g���R��G�C�3�g
=@2�\�XQ��33CG�                                    Bx�zU�  �          @�G��e@XQ��A���(�C�3�e@�H�u�&33C�                                    Bx�zd�  �          @��a�@`���-p���ffC0��a�@(Q��dz����CO\                                    Bx�zs2  �          @��R�b�\@_\)�4z����Cff�b�\@%��j�H��RC�
                                    Bx�z��  �          @�{�n{@W��+����HC	���n{@ ���`  �C�                                    Bx�z�~  �          @��R�tz�@s�
����z�C��tz�@J�H�3�
���
CO\                                    Bx�z�$  �          @��
�l(�@l(��
=��  C��l(�@>�R�A��\)C�q                                    Bx�z��  �          @��R�fff@\(��0����p�CO\�fff@$z��e����C��                                    Bx�z�p  �          @��
�s�
@s�
��\��  C���s�
@C33�N�R���CG�                                    Bx�z�  �          @�(��g�@l(��2�\����Cp��g�@333�k��33C:�                                    Bx�zټ  �          @�����  @g��
�H����C	�{��  @:=q�C�
��ffC�q                                    Bx�z�b  �          @���33@`�׿��H���RC���33@7
=�5���=qC��                                    Bx�z�  �          @�p���=q@Mp�������=qC����=q@$z��/\)�ܣ�C�)                                    Bx�{�  �          @�p���z�@C�
��\����C����z�@=q�1����
C�=                                    Bx�{T  �          @�(���p�@N{�	�����
C�f��p�@"�\�;����C#�                                    Bx�{"�  �          @��
��
=@Q�������C33��
=@!G��K���C!H                                    Bx�{1�  T          @��
���@K��!���{C����@���Q���RCxR                                    Bx�{@F  �          @��
���@I�������p�C�\���@���L(��C�                                    Bx�{N�  �          @��
���@Fff�=q��Q�C�{���@
=�H���z�C�{                                    Bx�{]�  �          @����ff@>�R�����\C5���ff@���C�
��(�C�                                    Bx�{l8  �          @�33��p�@>�R�=q�¸RC�q��p�@  �Fff� 33C                                      Bx�{z�  �          @��\����@C�
�!G����Cc�����@33�N�R���C�3                                    Bx�{��  �          @�G���z�@Fff�%���{C&f��z�@��S33�
=C��                                    Bx�{�*  �          @�����=q@E��,(���  C޸��=q@�\�X���  C�3                                    Bx�{��  �          @�Q����@AG��,(���=qC�����@�R�W��Q�C�                                     Bx�{�v  �          @�  ����@A��'
=��33C������@���S33���Cu�                                    Bx�{�  �          @������@E��,(��߮C������@�\�X����HCL�                                    Bx�{��  �          @�\)�|��@C�
�333���C0��|��@  �_\)�  CW
                                    Bx�{�h  �          @�Q���Q�@=p��   ��CB���Q�@{�J�H��Cu�                                    Bx�{�  �          @���Q�@1G��!���  C����Q�@�\�I�����Cp�                                    Bx�{��  �          @�
=��33@,(��#�
�ԣ�CJ=��33?����J=q���C�\                                    Bx�|Z  �          @�\)���H@,(��#�
��G�C8R���H?��H�J=q�=qC�q                                    Bx�|   �          @��\���\@#�
�;���p�Cc����\?�  �^�R��C�                                    Bx�|*�  �          @�����@���7
=���C.���?�{�W����CǮ                                    Bx�|9L  �          @��R��@��,����C#���?�\)�H���33C!�=                                    Bx�|G�  �          @���qG�@P���5���RC.�qG�@���c33�33C�q                                    Bx�|V�  �          @�\)�s�
@S33�.�R��C{�s�
@!G��]p����C��                                    Bx�|e>  �          @��\�n{@L���HQ���CQ��n{@��tz��${C�q                                    Bx�|s�  �          @��H�h��@QG��K��(�C
��h��@Q��x���'33C�=                                    Bx�|��  �          @�Q��w�@B�\�?\)���C�)�w�@p��i�����C@                                     Bx�|�0  �          @����qG�@J�H�@  ��
=C�3�qG�@�l(��
=C8R                                    Bx�|��  �          @�
=�s�
@L���O\)�=qC��s�
@�
�z�H�%z�C��                                    Bx�|�|  �          @�Q�����@G
=�N{�z�CE����@�R�xQ�� �RC�R                                    Bx�|�"  �          @�\)��G�@L(��B�\��z�C����G�@
=�n�R��C�3                                    Bx�|��  �          @�z��vff@_\)�-p���Q�C	ٚ�vff@.�R�^�R�ffC��                                    Bx�|�n  �          @�  �tz�@}p�������C��tz�@Q��Mp��=qCc�                                    Bx�|�  �          @���e�@����33���HCp��e�@qG��8����\C�=                                    Bx�|��  �          @��
�_\)@�\)��z��^�HB��q�_\)@����(���p�C�R                                    Bx�}`  �          @���g
=@�Q��z�����CE�g
=@k��7����HCp�                                    Bx�}  �          @��H�XQ�@��
�	����C^��XQ�@_\)�Dz��(�C
                                    Bx�}#�  �          @����aG�@c33�7
=��{C�3�aG�@1G��g���
C�q                                    Bx�}2R  �          @�=q�Vff@����   ��(�C�R�Vff@Tz��XQ����C8R                                    Bx�}@�  �          @��\�S33@��R�G���Q�C ��S33@c33�L�����C��                                    Bx�}O�  �          @����l��@z�H�33����CO\�l��@U��:�H��G�C
                                      Bx�}^D  �          @����mp�@�  ��p���  C�{�mp�@Z�H�7����HC	J=                                    Bx�}l�  �          @�=q�a�@���G����RC��a�@^{�:=q��Cu�                                    Bx�}{�  �          @�33�4z�@�33�!G���{B��)�4z�@�녿�ff��\)B�R                                    Bx�}�6  �          @�z��Dz�@�(���\)�/�B�Q��Dz�@���{��B��R                                    Bx�}��  �          @����XQ�@��ÿ����B�u��XQ�@����+����
C\                                    Bx�}��  �          @���^{@�Q�����{
=B�#��^{@����&ff��ffC��                                    Bx�}�(  �          @�ff�Z�H@��
��  �j�\B�
=�Z�H@����!���33C}q                                    Bx�}��  T          @��`  @�ff��\����C 0��`  @{��/\)��{C�R                                    Bx�}�t  �          @�p��~{@e�(���  C	�H�~{@<(��L�����C��                                    Bx�}�  �          @��
��  @�
�Fff� G�Cp���  ?�ff�_\)��C#�                                    Bx�}��  �          @��\���@J=q�(���  CW
���@!G��E� =qCE                                    Bx�}�f  �          @�=q�hQ�@��
�����*�RC�q�hQ�@�Q������\C#�                                    Bx�~  �          @�p��U�@�z�#�
��Q�B��{�U�@�Q쿎{�-�B�                                      Bx�~�  �          @�z��J=q@�
=���
�uB��f�J=q@��H����2�RB�L�                                    Bx�~+X  �          @��R�Mp�@�  >8Q�?��
B�aH�Mp�@���p�����B�L�                                    Bx�~9�  �          @���dz�@���>u@�
B�=q�dz�@�\)�W
=� z�B�                                    Bx�~H�  �          @�\)�n�R@�p�<�>�=qC ���n�R@�����G���
CO\                                    Bx�~WJ  �          @�  �k�@��R���H��\)B����k�@�\)���
�k�
C\)                                    Bx�~e�  �          @���g�@�\)�!G���G�B�Ǯ�g�@�\)��z����RC �                                    Bx�~t�  �          @���p��@�(��\)���\C��p��@��ͿǮ�q�C�                                    Bx�~�<  �          @����k�@�  �����HB�k��k�@�����333C ��                                    Bx�~��  �          @����h��@���>�ff@�Q�B��=�h��@�Q�(�����B�                                    Bx�~��  �          @�ff�j=q@��?E�@�C p��j=q@�p���  ���C �                                    Bx�~�.  �          @����p��@�
=?W
=A�C\�p��@�G������C�{                                    Bx�~��  T          @�z��p��@�?aG�A
ffCO\�p��@��׽��Ϳp��C�                                    Bx�~�z  �          @���aG�@�=q?���A�z�C33�aG�@��?#�
@�G�B�W
                                    Bx�~�   �          @�G��\��@�=q?��HA��C ���\��@��H?=p�@�B���                                    Bx�~��  �          @�(��e�@�ff?��AS�C �
�e�@�(�>�Q�@`  B�k�                                    Bx�~�l  �          @���\��@���?�\)AXz�B���\��@��R>�p�@i��B�p�                                    Bx�  �          @��H�K�@�z�?���A}p�B��)�K�@�(�?�@��
B�B�                                    Bx��  �          @���[�@��\?�p�A�p�C \)�[�@�33?G�@��RB�ff                                    Bx�$^  �          @�{�vff@��R>�@�C�\�vff@��R�   ���C�
                                    Bx�3  �          @�{�s�
@��?
=@��CB��s�
@��׾�p��c�
C)                                    Bx�A�  �          @�p��{�@��H?^�RA�C&f�{�@�p��u�\)C�
                                    Bx�PP  �          @����@����\)�0  C+���@�Q쿓33�3
=C:�                                    Bx�^�  �          @����q�@n�R�8����=qCW
�q�@C33�fff�=qC{                                    Bx�m�  �          @��
���@n�R�)����p�C	n���@E�W
=���C��                                    Bx�|B  T          @Å�\)@mp��.�R��33C	)�\)@C�
�\(��	�C�                                     Bx���  �          @\��\)@j�H������G�C����\)@Mp��#�
��\)C\)                                    Bx���  �          @�����{@fff��  �?33C����{@P�׿��H���C0�                                    Bx��4  �          @�G���{@c�
���
�D(�C� ��{@N�R��p�����Cs3                                    Bx���  �          @�  ��  @dz��R��
=C���  @W�����N�RC��                                    Bx�ŀ  �          @�����ff@N�R������  CxR��ff@2�\�{�¸RCL�                                    Bx��&  �          @�  ��  @Z=q��  �@z�CG���  @E��z����C�                                    Bx���  �          @�Q�����@\�Ϳ��
���RCǮ����@AG��(�����C^�                                    Bx��r  �          @�G�����@J=q�����C�\����@)���1G����HCB�                                    Bx��   �          @�����
@L(���
���C\)���
@-p��)������C�\                                    Bx���  �          @�\)��z�@8Q�������RCu���z�@p��������CQ�                                    Bx��d  �          @����p�@0�׿��H���C���p�@z��{�îCǮ                                    Bx��,
  �          @�
=��z�@+��ff��\)C=q��z�@p��%��Q�C��                                    Bx��:�  �          @�Q���=q@6ff�
=q���HCk���=q@��+���=qC�)                                    Bx��IV  �          @�  ���H@^�R�(���(�CO\���H@>�R�5���Q�C��                                    Bx��W�  �          @�������@9���
�H��G�C�H����@�H�,(����CG�                                    Bx��f�  �          @�\)���@(�����\)C�����?��H�*=q���
C!\)                                    Bx��uH  �          @�{���@=q�ff��33C�����?�z��1G��߮C�\                                    Bx����  �          @�
=��  @.�R�
=����C
��  @{�6ff��  C�                                    Bx����  �          @�{��
=?����z���ffC#  ��
=?�\)�%����C'��                                    Bx���:  �          @��
���@�������C�����?��H�-p���  C �\                                    Bx����  �          @��
����?�ff�z����RC#8R����?�=q�$z��У�C((�                                    Bx����  �          @����{?�Q������=qC$����{?z�H�   ��(�C)T{                                    Bx���,  �          @����33?�녿޸R���C%k���33?��
��p���\)C)�                                    Bx����  �          @�G�����?��
��\��(�C#޸����?�����C'��                                    Bx���x  �          @��\��\)?�������Up�C%0���\)?��˅�~{C'�3                                    Bx���  �          @�����{>�G���{����C/c���{=��
��z����HC3&f                                    Bx���  �          @������H?�{� ����z�C$E���H?^�R�.{���C)�{                                    Bx��j  �          @�����H@��C�
���\C�R���H?�(��\(��z�CW
                                    Bx��%  �          @��H��=q@=q�E� �C޸��=q?���_\)��HC}q                                    Bx��3�  �          @��\����@���G���RC������?�\�`���Q�C�{                                    Bx��B\  T          @�  ��Q�@��@  ���\C�f��Q�?��
�XQ���HCY�                                    Bx��Q  �          @�  �|(�@!G��S�
�{CT{�|(�?�\)�mp��!�RC��                                    Bx��_�  �          @����u@G
=�?\)��33C�u@ ���`  ��C�
                                    Bx��nN  �          @���`��@n�R�#�
�ԣ�C@ �`��@L���L����HC	��                                    Bx��|�  �          @�{�U@c�
�2�\���C0��U@?\)�X���(�C
#�                                    Bx����  �          @�
=�b�\@1G��b�\�
=C�f�b�\@��\)�0�CxR                                    Bx���@  �          @�{�{�?�p��dz��
=C:��{�?���w��,{C!��                                    Bx����  �          @�(��~�R?�Q��fff��C�R�~�R?��\�vff�,z�C%��                                    Bx����  �          @���|��?�33�q��(
=C u��|��?333�~{�2��C)��                                    Bx���2  �          @�  �}p�@AG��1�����C���}p�@�R�QG���C                                    Bx����  �          @�33�z�H@a�� ���ˮC	�3�z�H@A��Fff� �CE                                    Bx���~  �          @��H�z=q@qG��Q����C�z=q@U�1G���ffC��                                    Bx���$  T          @�=q�s�
@s33�����  C
=�s�
@W
=�5��C
��                                    Bx�� �  �          @�G��s�
@j�H��
����C{�s�
@Mp��:�H��33C�                                    Bx��p  �          @����l��@|�����G�C!H�l��@aG��0  ��\Ch�                                    Bx��  �          @���o\)@vff�����C+��o\)@\(��*�H�݅C	h�                                    Bx��,�  �          @����s33@�=q���R�pz�C  �s33@p  �����CY�                                    Bx��;b  �          @�\)�_\)@��������`(�C\�_\)@�  �
=���C!H                                    Bx��J  �          @�p��fff@������[33C޸�fff@w
=��\���RC�3                                    Bx��X�  �          @�(��N{@�33��Q��g�B�
=�N{@�
=����:�HB��{                                    Bx��gT  �          @�=q�S�
@��R��
=���HB�33�S�
@�=q��
=�C\)B��                                    Bx��u�  �          @��R�E�@�\)=�?��B��H�E�@�{�#�
��G�B�k�                                    Bx����  �          @���E@�G�>Ǯ@�ffB�k��E@�����33�o\)B�aH                                    Bx���F  �          @�(��J�H@��\>Ǯ@��B���J�H@��\��33�l(�B�z�                                    Bx����  �          @�(��<��@�
=>��H@�  B��q�<��@����=q�6ffB��                                    Bx����  �          @���B�\@��=#�
>��B�{�B�\@�33�333���B�Ǯ                                    Bx���8  �          @�p��:=q@��=���?��
B��:=q@�Q�+���=qB��{                                    Bx����  �          @���C�
@��H�����
B�L��C�
@��ÿ@  �{B��                                    Bx��܄  �          @�z��3�
@���>\@��B����3�
@��þ����n�RB���                                    Bx���*  �          @���@��@�{>��
@`��B�L��@��@�{�\���RB�aH                                    Bx����  �          @�=q�e@{���  �-p�Cff�e@u�c�
��\C�                                    Bx��v  �          @�33�g
=@z=q�u�$z�C�g
=@tz�aG��33Cp�                                    Bx��  �          @��R�j=q@mp�<��
>.{C�{�j=q@j�H�
=��  C�3                                    Bx��%�  �          @�z��z=q@l(��W
=�p�C�f�z=q@g
=�L���	��C	L�                                    Bx��4h  �          @����u@h�þ����b�\C�\�u@a녿h����\C	\)                                    Bx��C  �          @�{�n{@e��(�����C��n{@^{��  �2�RC�3                                    Bx��Q�  T          @�\)��G�@S33�����U�C����G�@Mp��Tz��=qC�                                     Bx��`Z  T          @�33��p�@S33�Ǯ��ffC�3��p�@L(��h���C�
                                    Bx��o   �          @�����@W
=��\)�@  C�����@S�
����У�C.                                    Bx��}�  �          @��\����@J�H���R�S33Ck�����@E��O\)�
=C33                                    Bx���L  �          @����p��@g
=>�33@xQ�C&f�p��@g��aG���C\                                    Bx����  �          @�Q��qG�@J=q�\)��\)C��qG�@Fff�!G����C�H                                    Bx����  �          @��\)@�ÿ��˅C)�\)@녿c�
�.�RCL�                                    Bx���>  �          @�z��xQ�@"�\�u�@  C�q�xQ�@{�#�
��z�Cu�                                    Bx����  �          @�  �hQ�@#33��=q�]p�C�H�hQ�@�R�(���	�C�f                                    Bx��Պ  �          @�=q��{@�  ?B�\AG�BϸR��{@��=u?+�B�\)                                    Bx���0  �          @�z�У�@�  >�ff@�p�B�  �У�@��׾W
=�'�B��
                                    Bx����  �          @�ff�J=q@��\?�R@�\)Bȳ3�J=q@��
�u�8Q�BȀ                                     Bx��|  �          @�����@�33>��@��B�W
���@������W�B�L�                                    Bx��"  �          @��\��@c33=L��?+�B�\��@`�׾��H��(�B�                                     Bx���  �          @w���\@Z�H=�\)?z�HBꞸ��\@X�þ�ff�أ�B�
=                                    Bx��-n  �          @\)��@\(���Q쿪=qB�aH��@X�ÿ(���\B�#�                                    Bx��<  �          @7
=���@���B�\�qG�B�B����@����0��B�z�                                    Bx��J�  �          @Z�H��@+������33B��=��@%�O\)�Z{C �q                                    Bx��Y`  �          @w���@@  �
=�z�B��=��@8Q쿃�
�xQ�C �                                    Bx��h  �          @w
=� ��@4z�!G���C��� ��@,�Ϳ���|��C�R                                    Bx��v�  �          @���@>�R?���A��RC8R��@G
=?+�A�RB���                                    Bx���R  �          @�{���@c33@33A���B�#����@s�
?�G�A�=qB�\                                    Bx����  �          @�  ��@n{?�\)A�=qB�Q���@z=q?�ffAN=qB�                                    Bx����  �          @�z��4z�@W�?���A��
C��4z�@b�\?p��A;33C �                                    Bx���D  �          @�G����@fff?��
AUG�B�u����@mp�?   @�33B��
                                    Bx����  �          @����@z=q?Tz�A)�B�G���@\)>��@P  B�L�                                    Bx��ΐ  �          @��\�=q@s33?L��A!�B��=q@xQ�>u@Dz�B�q                                    Bx���6  �          @�(���\@z=q?fffA4(�B�R��\@�  >���@��\B                                    Bx����  �          @�33�=q@p��?���AXz�B�u��=q@w�?�@љ�B��H                                    Bx����  �          @�
=�$z�@s33?���AQ�B���$z�@z=q?�\@ȣ�B��\                                    Bx��	(  �          @���p�@z=q?�ffARffB����p�@���>�@��Bힸ                                    Bx���  �          @�p���\@r�\?}p�AN�\B���\@x��>�G�@�ffB�B�                                    Bx��&t  �          @��
�  @~{?�R@�Q�B�(��  @���=L��?+�B�{                                    Bx��5  �          @��
�G�@{�?@  A�RB��H�G�@�  >B�\@Q�B�
=                                    Bx��C�  �          @��\�=q@n�R��G��MG�B��=q@c33���
��z�B�\)                                    Bx��Rf  �          @�p��!G�@w
=���
���B�Q��!G�@tz�\)��\)B��H                                    Bx��a  �          @���6ff@w
=>.{?��RB��H�6ff@u��p���{B��                                    Bx��o�  �          @����=p�@^{��ff�}�Cu��=p�@P�׿�\����CB�                                    Bx��~X  �          @�\)�=p�@a녿��H����C�q�=p�@P������z�C=q                                    Bx����  �          @�=q���@#�
>B�\@
=qC�{���@#�
�#�
����C�\                                    Bx����  �          @�=q��  ?��H�����p�C =q��  ?��\���
C ��                                    Bx���J  �          @�����
?�  ��(��x(�C�����
?��ÿ�z���ffC"33                                    Bx����  �          @�=q��=q?}p��޸R����C&Y���=q?=p���{���
C)��                                    Bx��ǖ  �          @��R�w
=?�z�������C�w
=?�{����p�C �{                                    Bx���<  �          @����x��@G���=q���C���x��?�  ��иRC�=                                    Bx����  T          @�������@333��\)����C�R����@#�
��p����CW
                                    Bx���  �          @�=q���\@B�\��  ��p�CE���\@4z�����\)Cc�                                    Bx��.  �          @�\)�l��@R�\��z����C
Q��l��@B�\�����
C�\                                    Bx���  �          @�
=�n{@XQ쿨���k\)C	�q�n{@K���G���  C��                                    Bx��z  �          @�ff�^{@h�ÿ�  �^�\C���^{@\�Ϳ�(����RC33                                    Bx��.   �          @����]p�@u��h��� z�C!H�]p�@k���z��yp�CJ=                                    Bx��<�  �          @����S�
@\)�^�R���C���S�
@u����t��C�\                                    Bx��Kl  �          @���7
=@�\)��R��B�8R�7
=@�������O�
B���                                    Bx��Z  �          @�p��=p�@�\)��{�g
=B�  �=p�@�z�k��G�B���                                    Bx��h�  �          @����@�  ��Q��y��B�=q��@���z�H�(z�B��                                    Bx��w^  �          @���0��@�(���\)�Dz�B�p��0��@�녿W
=�p�B�L�                                    Bx���  �          @����+�@��<��
>�  B�Ǯ�+�@�ff������HB�(�                                    Bx����  T          @����Q�@�p�?��@�
=B�#��Q�@�ff���
�W
=B��R                                    Bx���P  �          @��R�:=q@��\?   @���B��)�:=q@����Q�k�B�\                                    Bx����  �          @�G��R�\@��Ϳz���G�B�u��R�\@�G�����=�B��)                                    Bx����  �          @�z��Tz�@����(���p�B��)�Tz�@��Ϳ�  �$  B���                                    Bx���B  �          @�p��@��@��Ϳ�{�6{B��f�@��@�
=��Q���p�B���                                    Bx����  T          @�p��>�R@�z῝p��H��B�Q��>�R@�ff��ff���RB��                                    Bx���  �          @�  �?\)@�Q쿓33�9�B�8R�?\)@��\�޸R��p�B�B�                                    Bx���4  T          @��I��@�G���ff�S33B�� �I��@�33������ffB��f                                    Bx��	�  �          @����?\)@�\)��Q�����B�W
�?\)@����R��(�B�W
                                    Bx���  �          @��J=q@���޸R����B��J=q@���G���G�B�                                      Bx��'&  �          @��aG�@��������7�CT{�aG�@��
��33��\)Cz�                                    Bx��5�  T          @�\)�aG�@�
=����zffC�{�aG�@�  �33��{C\)                                    Bx��Dr  �          @�{�k�@�33��Q��jffC��k�@x�ÿ�
=����Cff                                    Bx��S  �          @���p  @�z῝p��F�\C5��p  @}p���p���ffC}q                                    Bx��a�  �          @�\)�W
=@�ff��G��L  B�\�W
=@�Q�����=qC 8R                                    Bx��pd  �          @�  �G
=@�{����  B�  �G
=@�{���33B�.                                    Bx��
  �          @��
�<(�@hQ��8Q���G�C�<(�@P  �R�\��RC�                                    Bx����  �          @��R�-p�@���G�����B�(��-p�@~{� ������B���                                    Bx���V  �          @��>�@��\�k��'
=B�>�@�Q�W
=��B���                                    Bx����  T          @��;�Q�@�33    ��B�(���Q�@�녿(��ۅB�8R                                    Bx����  �          @��H��p�@�=q�+���RBԅ��p�@�ff��G��\  B�=q                                    Bx���H  �          @��
����@��R�������B�녿���@�������G�B��                                    Bx����  �          @����r�\@h������Q�C.�r�\@U��1���
=C
�                                    Bx���  �          @���g
=@w
=�=q�ď\C��g
=@c33�6ff��G�C��                                    Bx���:  �          @��\�w�@]p��%��{C
.�w�@HQ��>�R��
=C�q                                    Bx���  �          @�\)�c33@b�\�/\)��\)C&f�c33@L(��HQ���C
{                                    Bx���  �          @�ff�`��@Q��B�\�p�C	��`��@9���Y���{CxR                                    Bx�� ,  �          @�{�]p�@;��Tz���C�q�]p�@!��hQ��!(�C�)                                    Bx��.�  �          @���Z=q@���?fffA
=C��Z=q@�(�>�(�@�33C�
                                    Bx��=x  �          @�Q��!G�@�33�������B�=�!G�@�z���
���HB���                                    Bx��L  �          @����\@�(���{���RB�33��\@�p��
=q��(�B�8R                                    Bx��Z�  �          @��R�{@����
=���\B��f�{@�{������B��                                    Bx��ij  T          @����<��@e�{��=qCn�<��@R�\�7����C�                                    Bx��x  �          @�z��C33@n{��p���{C^��C33@`  ������
C�                                    Bx����  �          @����,(�@~�R�����
B���,(�@r�\��{���\B��                                    Bx���\  �          @�{�G�@�  ��  �;�
B�33�G�@����(���p�B���                                    Bx���  F          @�Q���H@�G��
=�ə�B��H���H@qG��#33����B���                                    Bx����  	�          @�(����R��p�@~{B���C`�{���R�G�@��\B���CT(�                                    Bx���N  
Z          @�ff���>W
=@���B��\C-Y����?(��@~{B��{C�                                    Bx����  �          @��\��>�33@l��B|��C)���?B�\@h��BuffCs3                                    Bx��ޚ  "          @�(��.�R?B�\@QG�BDffC$n�.�R?�\)@J=qB<�C��                                    Bx���@  �          @��R��
=?�(�@e�BfQ�C�q��
=?���@[�BW\)C8R                                    Bx����  "          @���Vff?�p�?�\)A�{C���Vff?�\)?�A�(�C                                    Bx��
�  
�          @�33�?\)?���@#�
B�C���?\)@@B��C�                                    Bx��2  �          @�Q��.{=���@X��BM{C1��.{>��@W
=BJz�C*=q                                    Bx��'�  �          @����~{?�
=?�
=A�Q�C��~{?�{?��RA�  C�                                    Bx��6~  �          @�
=��  @z�?\(�A(  C�
��  @��?��@�\C                                      Bx��E$  
�          @�\)�y��@(Q�#�
���
C�y��@'
=��z��b�\C.                                    Bx��S�  �          @�Q����@p�>��@�C����@\)>.{@ ��C�                                    Bx��bp  
�          @��
����?�Q�?�(�Ax��C:�����?�?��
AQG�C�=                                    Bx��q  
(          @����  ?�{?У�A���C%Q���  ?��
?�  A�G�C#:�                                    Bx���  
�          @���~�R@z�?��
A33C�=�~�R@(�?�ffAP��C&f                                    Bx���b  T          @�  �}p�@�
?��ALQ�C���}p�@=q?J=qAffC��                                    Bx���  	�          @�����
�\)>k�@8Q�C5�����
��>�  @C33C5z�                                    Bx����  �          @�
=���=u�
=��  C3:����    �����G�C4                                      Bx���T  
�          @�ff���\�.{�5�33C<k����\�@  �!G�����C=J=                                    Bx����  T          @������R����{�^�RC:���R�#�
��ff�R=qC<33                                    Bx��נ  
(          @�p������=q���R�v�HC7aH����Ǯ���H�o�
C9�                                    Bx���F  
~          @�����>�������
C0����>W
=�����C1c�                                    Bx����  
�          @��\��33?�\)�����
C"� ��33?��;���Tz�C"�q                                    Bx���  
�          @�G���Q�?�����
��=qC 
��Q�?�  ����=qC �
                                    Bx��8  
�          @�p���Q�?��z���z�C���Q�?�  �B�\��HC�R                                    Bx�� �  
�          @�z�����?Ǯ�h���4  C������?��H����S�C!0�                                    Bx��/�  T          @����Q�?��h���8  C!���Q�?��ÿ���T��C"                                    Bx��>*  "          @���G�?���Tz��$z�C E��G�?�Q�z�H�C�C!aH                                    Bx��L�  "          @�p����H>������Q�C-�\���H>�����=q���RC/Ǯ                                    Bx��[v  T          @�=q��Q�?aG���(��z�\C(G���Q�?B�\��ff��(�C)�3                                    Bx��j  "          @�33��{?�{��Q���  C@ ��{?��ÿ
=q��\)C                                    Bx��x�  
�          @�33��Q�@�׾����xQ�C
��Q�@p�����ҏ\C�\                                    Bx���h  
(          @�  ��(�@�\��ff��{C����(�@\)�.{��C��                                    Bx���  "          @��R���@ff�#�
� ��C�3���@z������z�C@                                     Bx����  �          @�Q�����@�\�u�:ffC�\����@���
=�f�HC��                                    Bx���Z  	�          @�=q��G�@&ff>��
@q�C=q��G�@'
==u?(��C\                                    Bx���   �          @�p��s33@#�
?fffA2ffC�s33@(��?#�
@��C:�                                    Bx��Ц  �          @�\)�s�
@!G�?���A^�HC�=�s�
@'�?aG�A+�
C�                                     Bx���L  �          @�G��g�@=p�?O\)A��C�R�g�@A�?�\@�p�C�                                    Bx����  "          @�33�]p�@U=�\)?W
=C  �]p�@U�����G
=C{                                    Bx����  �          @��
�HQ�@b�\?���AV�\Cz��HQ�@hQ�?E�A=qC��                                    Bx��>  �          @�=q�N�R@HQ�?˅A�(�C���N�R@QG�?��
Ax��C��                                    Bx���  �          @����K�@%�@A�(�C�)�K�@333@�A�
=C
�{                                    Bx��(�  
�          @�z��	��@j�H?�(�A�B��
�	��@tz�?���A���B��)                                    Bx��70  
�          @�zᾨ��@��\?��@��HB�B�����@��=���?�  B�33                                    Bx��E�  
�          @���u@�G�?E�A\)B�LͿu@��H>�33@�  B�                                      Bx��T|  �          @�Q쿽p�@��?p��A@  B�33��p�@�\)?�@׮Bڔ{                                    Bx��c"  "          @����G�@���?n{AC
=B���G�@�33?�@�
=B�\)                                    Bx��q�  
�          @�=q��(�@n�R?^�RA:�HB�\��(�@s33?�\@�Q�B�R                                    Bx���n  	�          @�  ��{@���?��@�G�B�ff��{@��\>#�
@ ��B�
=                                    Bx���  
�          @��\�	��@~{>�{@��\B�q�	��@\)�#�
��B�\                                    Bx����  T          @����U�?���@  Bz�C E�U�?�z�@�A���C#�                                    Bx���`  �          @�=q�I��@33@�\A�{C��I��@\)?�=qA�G�C�)                                    Bx���  T          @���Vff?��@�A��
C�R�Vff?�p�?�33A��C:�                                    Bx��ɬ  �          @�33�.�R��G�@'�B-�C=\�.�R�8Q�@)��B0  C7޸                                    Bx���R  �          @��\��\)@��?��@�RB�uþ�\)@�z�>\)?�\B�ff                                    Bx����  �          @���@���>8Q�@��B���@��׾k��7�B�Ǯ                                    Bx����  �          @�p����H@��R?Y��A0Q�B�8R���H@���>�ff@���B�Ǯ                                    Bx��D  
�          @fff?�ff@.{?�A�{Bb(�?�ff@5?�33A��Bf                                    Bx���  
�          @Vff?�33@=p�=u?���B�L�?�33@=p��aG��p��B�33                                    Bx��!�  �          @3�
?��@#�
<�?
=qB���?��@#33�aG���B���                                    Bx��06  �          @�p�@1G�@Q녾������HBG33@1G�@O\)�!G��
=BE��                                    Bx��>�  �          @��R@~�R@녿0���z�A�z�@~�R?��H�aG��333A�                                    Bx��M�  
�          @��?�p�@=p�����
B}�H?�p�@9���@  �Pz�B|
=                                    Bx��\(  
�          @���{@��?L��Az�B��{@��
>�
=@�{B��f                                    Bx��j�  
�          @�Q��)��@��?^�RA�
B����)��@�
=>�@�
=B�33                                    Bx��yt  
�          @�\)�a�@Q�?��\A��HC�a�@\)?��AYG�C�
                                    Bx���  "          @��\�Fff@b�\?���AS�C&f�Fff@hQ�?E�A
=Cu�                                    Bx����  
�          @��
�"�\@|(�?���Ae�B����"�\@���?Tz�Ap�B�ff                                    Bx���f  
�          @�\)�(��@�(�?E�A\)B���(��@�>Ǯ@��B�p�                                    Bx���  
�          @��R�'
=@�ff>k�@(Q�B���'
=@��R�#�
��{B�                                    Bx��²  
�          @�\)�z�@���k��+�B�{�z�@�=q�!G����HB�                                     Bx���X  
�          @��R�   @�\)>�@��B���   @�Q�=�Q�?��\B���                                    Bx����  
�          @�z��'
=@mp�>�p�@�
=B�8R�'
=@n�R<�>��B��                                    Bx���  "          @�zῴz�@�=q�L���Q�B�33��z�@�\)��(��h��B��)                                    Bx���J  	�          @��\��(�@��H��
=����B�G���(�@�p��z���ffB��q                                    Bx���  "          @��Ϳ�ff@�z�?J=qA(�B�33��ff@�{>�Q�@��RB��H                                    Bx���  �          @����\)@��?�R@�{B�aH��\)@�33>L��@�B��                                    Bx��)<  �          @��
��z�@�ff=#�
?�B�uÿ�z�@�{��Q���Q�B�{                                    Bx��7�  
�          @��
��{@�{>��H@��HB���{@�
==�Q�?��
B�G�                                    Bx��F�  �          @��Ϳ�\)@��?B�\A  B�
=��\)@��>�{@�Q�BԳ3                                    Bx��U.  T          @�{�G�@j=q?���A���B�q�G�@r�\?�G�A~ffB���                                    Bx��c�  
�          @��
>�p�@�녿��R���\B���>�p�@����33���B��                                     Bx��rz  �          @�=q?u@��\��\)����B���?u@�z��  ��33B��f                                    Bx���   �          @�z�>�@���xQ��=��B�  >�@�  ������  B��R                                    Bx����  �          @��׿У�@���?��AUG�B�aH�У�@�(�?:�HA\)BڸR                                    Bx���l  �          @�
=�^�R@��H����(�B�k��^�R@�녿\)��G�BȊ=                                    Bx���  �          @�G�=u@��R?G�A�RB���=u@�Q�>���@qG�B���                                    Bx����  �          @�논�@�  ?.{@�{B��׼�@�G�>aG�@#33B���                                    Bx���^  �          @�  �\@�{>��H@��
B�녾\@��R=#�
?   B��)                                    Bx���  �          @�����Q�@�?!G�@�G�B�aH��Q�@�
=>B�\@
�HB�Q�                                    Bx���  �          @��Ϳ��>�p�@��
B��=C(�׿��?B�\@�=qB���C�                                    Bx���P  �          @�{���@1G�@s33BL
=B瞸���@Fff@b�\B9ffB�                                    Bx���  �          @��{@'
=@_\)B6\)Cs3�{@:=q@P  B&p�B�Ǯ                                    Bx���  �          @�33��{@s�
@G�A��B�(���{@�  ?�A�  B�u�                                    Bx��"B  �          @�{�5@�z�@ffA���B�G��5@��?��HA�ffBŞ�                                    Bx��0�  �          @�Q�^�R@��@�A�p�B�z�^�R@�G�?��A�\)Bɔ{                                    Bx��?�  �          @��\��G�@�\)=u?0��B���G�@�
=�\��z�B�\                                    Bx��N4  �          @��H��\@�{?G�A=qB�aH��\@��>�Q�@�=qB�B�                                    Bx��\�  �          @��H����@���?��
Aj=qB�\����@��?Y��A�HB͔{                                    Bx��k�  T          @�z����@���?��\A8  B�k�����@�33?��@�p�B��f                                    Bx��z&  �          @����=q@�ff?c�
A�HB̏\��=q@�Q�>�ff@�
=B�G�                                    Bx����  �          @�=q���@�?�Q�A�{Bҳ3���@�G�?��\A9�B�
=                                    Bx���r  �          @��\��{@�Q�?�p�A��
B�(���{@��
?�ffAB�RB���                                    Bx���  �          @���?��@��\?���Au�B�?��@�?h��A(Q�B��                                     Bx����  �          @��
>�z�@���@33A��B�>�z�@�=q?У�A�G�B�                                    Bx���d  �          @���=���@���@	��A�
=B�k�=���@��R?޸RA��\B��                                     Bx���
  �          @�Q�?L��@�@�\A���B��=?L��@��H?��A�
=B�=q                                    Bx���  �          @�\)?�@i��@AG�B�
B�aH?�@y��@,(�B�\B��=                                    Bx���V  �          @�>�=q@h��@N�RB&�B��\>�=q@z=q@9��B  B�#�                                    Bx����  �          @��>L��@W
=@eB;(�B�.>L��@j=q@Q�B'  B��q                                    Bx���  �          @��׿�G�@���@p�A�\)B�#׿�G�@�\)?���A�(�B���                                    Bx��H  �          @��R��\@?\)@5B��B�Ǯ��\@N�R@#�
B�RB���                                    Bx��)�  �          @��׿&ff@�
=?�=qAc
=B�k��&ff@���?333A33B��                                    Bx��8�  �          @�녿}p�@���?�=qA�\)B�=q�}p�@��?���A�  B�ff                                    Bx��G:  �          @��
>�33@�p�?�p�A��\B�
=>�33@���?��A�Q�B�L�                                    Bx��U�  �          @��׾Ǯ@��@
=qA�=qB��q�Ǯ@��H?��
A�{B�Q�                                    Bx��d�  �          @�Q��3�
@�\@>{Bp�C�H�3�
@"�\@0��BQ�C	�                                    Bx��s,  �          @�=q���?�=q@{�B\(�C
=���?�@s33BQ�C5�                                    Bx����  �          @��\���?˅@��B���B��\���?�(�@��RBw�B���                                    Bx���x  �          @��׿��
@3�
@�RB�B�녿��
@AG�@�RBG�B�{                                    Bx���  T          @�=q���@h��?�ffA�=qB��f���@n{?:�HA0z�B���                                    Bx����  �          @�(�?�Q�@�G���G��o�B�8R?�Q�@���У���Q�B��f                                    Bx���j  �          @�z�?���@��Ϳٙ����B��?���@\)��
��Q�B�(�                                    Bx���  �          @��?Ǯ@k��
=q��33B�#�?Ǯ@^�R�{�{B�                                    Bx��ٶ  �          @�33�n{@���
=q���
B��Ϳn{@�p��h���@z�B�(�                                    Bx���\  T          @�\)�j=q@2�\?^�RA+�
C���j=q@7
=?�R@��C                                    Bx���  �          @��H�QG�@`��>L��@��C���QG�@aG���G����C޸                                    Bx���  �          @�ff�dz�@6ff������Cp��dz�@2�\�:�H�ffC�R                                    Bx��N  �          @�{�`��>�33@'�B��C.T{�`��?z�@%�B  C*�f                                    Bx��"�  �          @�33�g
=?�33@�A��C5��g
=@�@   A͙�C�R                                    Bx��1�  �          @��׿�z�@{��
=q��G�B���z�@w
=�aG��9�B��                                    Bx��@@  T          @���\)@��������HB��{�\)@����33����B�#�                                    Bx��N�  �          @�Q�?�@��������B�\?�@�p�������B���                                    Bx��]�  T          @���>\)@��\��33�j�\B�>\)@�{��{����B��                                    Bx��l2  T          @�G���@�Q��
=��\)B��;�@��H�Q����B��f                                    Bx��z�  �          @��Ϳ�@��ͿE���\B�W
��@�녿����^�HB��\                                    Bx���~  �          @�녾���@�ff?�\)A�ffB�L;���@�=q?�
=APQ�B�{                                    Bx���$  �          @�녿333@���?B�\A�BĮ�333@��\>�{@~{Bą                                    Bx����  �          @���@��ÿУ���33B�p���@��
�z���p�B�Ǯ                                    Bx���p  
�          @��
����@�\)�@  �ffB�\����@�zῗ
=�O�Bף�                                    Bx���  �          @�G��
=@����
��=qB�8R�
=@���   ��z�B�Ǯ                                    Bx��Ҽ  �          @�=q�^�R@�ff��z���{B�녿^�R@�������  BǊ=                                    Bx���b  "          @�p����H@�33�^�R��B��H���H@�  �����c�
B�z�                                    Bx���  
�          @��R��{@��׿(����Bˮ��{@�ff�����5�B�                                    Bx����  �          @��þ��H@��׿�\)�h��B��R���H@�(���=q���HB�                                      Bx��T  "          @�Q쿃�
@��H��  ��
=B��)���
@�p�������HBˣ�                                    Bx���  �          @��׾aG�@�������F{B��
�aG�@�G��˅���B���                                    Bx��*�  �          @�G�>��@�녿��p  B�>��@�p�������RB��                                    Bx��9F  
(          @�Q��{@�{?�(�AP  B�8R��{@���?E�A�B܏\                                    Bx��G�  
�          @��
�'
=@��@A�33B���'
=@�\)?�(�A���B�ff                                    Bx��V�  �          @�����@N�R>�ff@�z�C#����@P��>#�
?�  C�f                                    Bx��e8  
Z          @����XQ�@P��@
=A��C�q�XQ�@\(�?�A��HC}q                                    Bx��s�  �          @�(��{@��R?�
=A���B��{@�33?�ffAn�\B�\                                    Bx����  T          @��   @�=q?��HAY�B�B��   @��?Q�A�HB�8R                                    Bx���*  �          @��R���@���?�R@�\)B�����@��H>W
=@
=B�B�                                    Bx����  "          @���`��@hQ콸Q�}p�C�q�`��@g
=�����C+�                                    Bx���v  �          @�G��z=q@<�Ϳ���D(�C  �z=q@5�����u�C                                    Bx���  T          @�G��^{@U��G���ffC��^{@R�\�=p��(�C�                                    Bx����  �          @����(�@!G��(����  C����(�@�ͿaG��&�\CG�                                    Bx���h  �          @����~�R@!녿�p�����C�\�~�R@
=��������CJ=                                    Bx���  "          @��R���?�(������NffC����?�{��  �k
=C B�                                    Bx����  T          @�ff���?������M�C'&f���?u��
=�^�RC(\)                                    Bx��Z  "          @�\)��\)>�
=��z����
C.����\)>�zῸQ�����C0��                                    Bx��   "          @�
=��������{�N�RC9k�������Ϳ���F{C:�{                                    Bx��#�  �          @������
>�ff�c�
�$z�C.�����
>�p��k��+�C/�f                                    Bx��2L  
�          @�G���ff?5>B�\@��C+�
��ff?:�H>�?���C+��                                    Bx��@�  �          @�����\)?�  <��
>��C����\)?�  �\)��ffC�R                                    Bx��O�  T          @�������@ff>L��@�C0�����@
=<#�
=�Q�C
                                    Bx��^>  T          @�ff��ff@��?�@\C5���ff@(�>�z�@]p�C�{                                    Bx��l�  T          @�  �e�@G
=?�p�A�=qC\�e�@N�R?�Q�A\��C	�R                                    Bx��{�  
�          @�  �>�R@^{?�
=A�G�C���>�R@h��?�{A��C^�                                    Bx���0  
�          @�\)�1G�@k�?˅A�Q�B��f�1G�@s�
?�  Am�B�                                      Bx����  
�          @������@��?333A�B�  ����@��>���@mp�B֨�                                    Bx���|  �          @��
����@��\>�@�
=BՏ\����@�33=#�
>�
=B�ff                                    Bx���"  T          @��
�(��@�zἣ�
��z�B�8R�(��@��
��ff����B�G�                                    Bx����  �          @��R���@�{��{��p�B�LͿ��@��ÿ��R��=qB���                                    Bx���n  �          @�G���(�@�녿�  �hz�B�.��(�@�p���Q���z�B�z�                                    Bx���  T          @��:�H@��ÿ�  ��\)BȊ=�:�H@w���\)���B�\)                                    Bx���  �          @�{@���>�(�@J�HBQ�@�p�@���?:�H@G
=BG�A#�
                                    Bx���`  �          @��H@�z�p��@,��B�HC��{@�z�0��@1G�B  C�G�                                    Bx��  T          @�33@�p�?�?�z�A��\@�\)@�p�?5?���A���A	p�                                    Bx���  �          @�  @��H?���@33A���A�Q�@��H?��?�z�A���A��H                                    Bx��+R  �          @���@��\@6ff?:�HA�B��@��\@9��>�@���B�                                    Bx��9�  �          @�=q?�33@����=q�E�B�#�?�33@�녿333� ��B��R                                    Bx��H�  
�          @�(�@-p�@~�R��z����HB^�@-p�@u����
��=qBZ��                                    Bx��WD  
Z          @��R@mp�@S33?���AL(�B&��@mp�@X��?O\)AffB)�                                    Bx��e�  
(          @�ff@j=q@Z=q?���AF�RB+�
@j=q@`  ?G�A\)B.��                                    Bx��t�  �          @�@g�@c33�(����z�B1�@g�@^{�}p��2�\B/�                                    Bx���6  T          @�\)@~�R@I����{�E��B�@~�R@A녿�z��z�HB=q                                    Bx����  	�          @�  @Y��@~{��Q��|(�BE@Y��@z�H�:�H� ��BDG�                                    Bx����  F          @�ff@R�\@z�H�E����BH=q@R�\@u�����K33BE�\                                    Bx���(  
6          @��@:=q@�녾�33�vffB_�
@:=q@�Q�B�\��B^�                                    Bx����  "          @��
@L��@�p���  �,(�BR�@L��@�녿���qG�BO
=                                    Bx���t  
�          @��H@=q@�G�����d  Bxff@=q@��Ϳ�p����RBuG�                                    Bx���  
�          @��@��@��׿�33�{33B�Q�@��@����=q��G�B�                                    